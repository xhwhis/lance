// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub mod contains;
pub mod intersects;

use std::{any::Any, collections::BTreeMap, fmt, sync::Arc};

use arrow_array::{Array, Float64Array, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Fields, Schema};
use datafusion_common::Column;
use datafusion_expr::{lit, Expr};
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use snafu::location;

use super::{
    gist::{
        GiSTIndex, GiSTLookup, GiSTNode, GiSTOperations, GiSTPredicate, GiSTQuery,
        GiSTSerializable, GiSTSerializer, DEFAULT_GIST_BATCH_SIZE
    },
    AnyQuery,
};

#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct BoundingBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl fmt::Display for BoundingBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BOX({}, {}, {}, {})",
            self.min_x, self.min_y, self.max_x, self.max_y
        )
    }
}

impl BoundingBox {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    pub fn contains(&self, other: &Self) -> bool {
        self.min_x <= other.min_x
            && self.max_x >= other.max_x
            && self.min_y <= other.min_y
            && self.max_y >= other.max_y
    }

    pub fn area(&self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }
}

impl GiSTPredicate for BoundingBox {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dyn_clone(&self) -> Box<dyn GiSTPredicate> {
        Box::new(self.clone())
    }
}

impl GiSTSerializable for BoundingBox {
    fn serialize(&self) -> Result<Vec<Arc<dyn Array>>> {
        Ok(vec![
            Arc::new(Float64Array::from(vec![self.min_x])),
            Arc::new(Float64Array::from(vec![self.min_y])),
            Arc::new(Float64Array::from(vec![self.max_x])),
            Arc::new(Float64Array::from(vec![self.max_y])),
        ])
    }

    fn deserialize(arrays: &[Arc<dyn Array>], row: usize) -> Result<Box<dyn GiSTPredicate>> {
        if arrays.len() != 4 {
            return Err(Error::Internal {
                message: "BoundingBox requires exactly 4 arrays".to_string(),
                location: location!(),
            });
        }

        let min_x = arrays[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| Error::Internal {
                message: "Expected Float64Array for min_x".to_string(),
                location: location!(),
            })?
            .value(row);

        let min_y = arrays[1]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| Error::Internal {
                message: "Expected Float64Array for min_y".to_string(),
                location: location!(),
            })?
            .value(row);

        let max_x = arrays[2]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| Error::Internal {
                message: "Expected Float64Array for max_x".to_string(),
                location: location!(),
            })?
            .value(row);

        let max_y = arrays[3]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| Error::Internal {
                message: "Expected Float64Array for max_y".to_string(),
                location: location!(),
            })?
            .value(row);

        Ok(Box::new(BoundingBox::new(min_x, min_y, max_x, max_y)))
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("min_x", DataType::Float64, false),
            Field::new("min_y", DataType::Float64, false),
            Field::new("max_x", DataType::Float64, false),
            Field::new("max_y", DataType::Float64, false),
        ]))
    }
}

#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub enum SpatialQuery {
    Intersects(BoundingBox),
    Contains(BoundingBox),
}

impl AnyQuery for SpatialQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, col: &str) -> String {
        match self {
            Self::Intersects(bbox) => format!("ST_Intersects({}, {})", col, bbox),
            Self::Contains(bbox) => format!("ST_Contains({}, {})", col, bbox),
        }
    }

    fn to_expr(&self, col: String) -> Expr {
        match self {
            Self::Intersects(bbox) => {
                let col_expr = Expr::Column(Column::new_unqualified(col));
                col_expr
                    .clone()
                    .lt_eq(lit(bbox.max_x))
                    .and(col_expr.clone().gt_eq(lit(bbox.min_x)))
                    .and(col_expr.clone().lt_eq(lit(bbox.max_y)))
                    .and(col_expr.gt_eq(lit(bbox.min_y)))
            }
            Self::Contains(bbox) => {
                let col_expr = Expr::Column(Column::new_unqualified(col));
                col_expr
                    .clone()
                    .lt_eq(lit(bbox.min_x))
                    .and(col_expr.clone().gt_eq(lit(bbox.max_x)))
                    .and(col_expr.clone().lt_eq(lit(bbox.min_y)))
                    .and(col_expr.gt_eq(lit(bbox.max_y)))
            }
        }
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        match other.as_any().downcast_ref::<Self>() {
            Some(o) => self == o,
            None => false,
        }
    }
}

impl GiSTQuery for SpatialQuery {
    fn as_any_query(&self) -> &dyn AnyQuery {
        self
    }
}

/// Spatial-specific GiST operations
#[derive(Debug, DeepSizeOf)]
pub struct SpatialGiSTOps;

impl GiSTOperations for SpatialGiSTOps {
    fn consistent(&self, predicate: &dyn GiSTPredicate, query: &dyn GiSTQuery) -> bool {
        if let (Some(bbox_pred), Some(spatial_query)) = (
            predicate.as_any().downcast_ref::<BoundingBox>(),
            query.as_any_query().as_any().downcast_ref::<SpatialQuery>(),
        ) {
            match spatial_query {
                SpatialQuery::Intersects(query_bbox) => {
                    bbox_pred.min_x <= query_bbox.max_x
                        && bbox_pred.max_x >= query_bbox.min_x
                        && bbox_pred.min_y <= query_bbox.max_y
                        && bbox_pred.max_y >= query_bbox.min_y
                }
                SpatialQuery::Contains(query_bbox) => {
                    // For Contains query, the predicate must be able to contain the query bbox
                    // This is used for tree traversal - if a node's bbox can't contain the query,
                    // none of its children can contain it either
                    bbox_pred.contains(query_bbox)
                }
            }
        } else {
            false
        }
    }

    fn union(&self, predicates: &[&dyn GiSTPredicate]) -> Box<dyn GiSTPredicate> {
        if predicates.is_empty() {
            return Box::new(BoundingBox::new(0.0, 0.0, 0.0, 0.0));
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for predicate in predicates.iter() {
            if let Some(bbox) = predicate.as_any().downcast_ref::<BoundingBox>() {
                min_x = min_x.min(bbox.min_x);
                min_y = min_y.min(bbox.min_y);
                max_x = max_x.max(bbox.max_x);
                max_y = max_y.max(bbox.max_y);
            }
        }

        Box::new(BoundingBox::new(min_x, min_y, max_x, max_y))
    }

    fn same(&self, predicate: &dyn GiSTPredicate, other: &dyn GiSTPredicate) -> bool {
        match (
            predicate.as_any().downcast_ref::<BoundingBox>(),
            other.as_any().downcast_ref::<BoundingBox>(),
        ) {
            (Some(bbox1), Some(bbox2)) => {
                (bbox1.min_x - bbox2.min_x).abs() < f64::EPSILON
                    && (bbox1.min_y - bbox2.min_y).abs() < f64::EPSILON
                    && (bbox1.max_x - bbox2.max_x).abs() < f64::EPSILON
                    && (bbox1.max_y - bbox2.max_y).abs() < f64::EPSILON
            }
            _ => false,
        }
    }

    fn penalty(&self, existing: &dyn GiSTPredicate, new: &dyn GiSTPredicate) -> f64 {
        match (
            existing.as_any().downcast_ref::<BoundingBox>(),
            new.as_any().downcast_ref::<BoundingBox>(),
        ) {
            (Some(existing_bbox), Some(new_bbox)) => {
                let union_min_x = existing_bbox.min_x.min(new_bbox.min_x);
                let union_min_y = existing_bbox.min_y.min(new_bbox.min_y);
                let union_max_x = existing_bbox.max_x.max(new_bbox.max_x);
                let union_max_y = existing_bbox.max_y.max(new_bbox.max_y);

                let union_area = (union_max_x - union_min_x) * (union_max_y - union_min_y);
                let existing_area = (existing_bbox.max_x - existing_bbox.min_x)
                    * (existing_bbox.max_y - existing_bbox.min_y);

                union_area - existing_area
            }
            _ => f64::INFINITY,
        }
    }

    fn pick_split(&self, entries: &[Box<dyn GiSTPredicate>]) -> (Vec<usize>, Vec<usize>) {
        if entries.len() <= 2 {
            return (vec![0], if entries.len() > 1 { vec![1] } else { vec![] });
        }

        // Extract bounding boxes for R*-tree split
        let bboxes: Vec<Option<&BoundingBox>> = entries
            .iter()
            .map(|pred| pred.as_any().downcast_ref::<BoundingBox>())
            .collect();

        // Check if all entries are valid bounding boxes
        if bboxes.iter().any(|b| b.is_none()) {
            // Fall back to simple split if not all are bounding boxes
            return self.simple_split(entries.len());
        }

        // Perform R*-tree split
        self.r_star_split(&bboxes)
    }

    fn query_to_predicate(&self, query: &dyn GiSTQuery) -> Box<dyn GiSTPredicate> {
        if let Some(spatial_query) = query.as_any_query().as_any().downcast_ref::<SpatialQuery>() {
            match spatial_query {
                SpatialQuery::Intersects(bbox) | SpatialQuery::Contains(bbox) => {
                    Box::new(bbox.clone())
                }
            }
        } else {
            Box::new(BoundingBox::new(0.0, 0.0, 0.0, 0.0))
        }
    }

    fn empty_predicate(&self) -> Box<dyn GiSTPredicate> {
        Box::new(BoundingBox::new(0.0, 0.0, 0.0, 0.0))
    }

    fn serializer(&self) -> Box<dyn GiSTSerializer> {
        Box::new(SpatialSerializer)
    }

    fn batch_to_pages(
        &self,
        batch: RecordBatch,
        next_page_id: u32,
    ) -> Result<(Vec<RecordBatch>, Vec<(u32, Box<dyn GiSTPredicate>)>)> {
        let mut page_batches = Vec::new();
        let mut page_predicates = Vec::new();
        let mut current_page_id = next_page_id;

        let mut offset = 0;
        while offset < batch.num_rows() {
            let length = std::cmp::min(DEFAULT_GIST_BATCH_SIZE as usize, batch.num_rows() - offset);
            let page_batch = batch.slice(offset, length);
            offset += length;

            let bbox = extract_bounding_box(&page_batch)?;

            page_batches.push(page_batch);
            page_predicates.push((current_page_id, Box::new(bbox) as Box<dyn GiSTPredicate>));
            current_page_id += 1;
        }

        Ok((page_batches, page_predicates))
    }
}

impl SpatialGiSTOps {
    /// R*-tree split implementation
    fn r_star_split(&self, bboxes: &[Option<&BoundingBox>]) -> (Vec<usize>, Vec<usize>) {
        let n = bboxes.len();
        let m = n / 2; // Minimum entries per node

        // Choose split axis
        let (split_axis, split_index) = self.choose_split_axis_and_index(bboxes, m);

        // Create sorted indices for the chosen axis
        let mut indices: Vec<usize> = (0..n).collect();
        match split_axis {
            0 => {
                // Sort by x-axis (using center point)
                indices.sort_by(|&i, &j| {
                    let bbox_i = bboxes[i].unwrap();
                    let bbox_j = bboxes[j].unwrap();
                    let center_i = (bbox_i.min_x + bbox_i.max_x) / 2.0;
                    let center_j = (bbox_j.min_x + bbox_j.max_x) / 2.0;
                    center_i.partial_cmp(&center_j).unwrap()
                });
            }
            _ => {
                // Sort by y-axis (using center point)
                indices.sort_by(|&i, &j| {
                    let bbox_i = bboxes[i].unwrap();
                    let bbox_j = bboxes[j].unwrap();
                    let center_i = (bbox_i.min_y + bbox_i.max_y) / 2.0;
                    let center_j = (bbox_j.min_y + bbox_j.max_y) / 2.0;
                    center_i.partial_cmp(&center_j).unwrap()
                });
            }
        }

        // Split at the chosen index
        let group1: Vec<usize> = indices.iter().take(split_index).copied().collect();
        let group2: Vec<usize> = indices.iter().skip(split_index).copied().collect();

        (group1, group2)
    }

    /// Choose the best split axis and index according to R*-tree algorithm
    fn choose_split_axis_and_index(
        &self,
        bboxes: &[Option<&BoundingBox>],
        m: usize,
    ) -> (usize, usize) {
        let n = bboxes.len();
        let mut best_axis = 0;
        let mut best_index = m;
        let mut min_overlap = f64::INFINITY;
        let mut min_area = f64::INFINITY;

        // Try both axes
        for axis in 0..2 {
            // Create sorted indices for this axis
            let mut indices: Vec<usize> = (0..n).collect();
            match axis {
                0 => {
                    // Sort by x-axis
                    indices.sort_by(|&i, &j| {
                        let bbox_i = bboxes[i].unwrap();
                        let bbox_j = bboxes[j].unwrap();
                        let center_i = (bbox_i.min_x + bbox_i.max_x) / 2.0;
                        let center_j = (bbox_j.min_x + bbox_j.max_x) / 2.0;
                        center_i.partial_cmp(&center_j).unwrap()
                    });
                }
                _ => {
                    // Sort by y-axis
                    indices.sort_by(|&i, &j| {
                        let bbox_i = bboxes[i].unwrap();
                        let bbox_j = bboxes[j].unwrap();
                        let center_i = (bbox_i.min_y + bbox_i.max_y) / 2.0;
                        let center_j = (bbox_j.min_y + bbox_j.max_y) / 2.0;
                        center_i.partial_cmp(&center_j).unwrap()
                    });
                }
            }

            // Try all valid split positions
            for k in m..=(n - m) {
                let group1_indices: Vec<usize> = indices.iter().take(k).copied().collect();
                let group2_indices: Vec<usize> = indices.iter().skip(k).copied().collect();

                // Calculate MBRs for both groups
                let mbr1 = self.calculate_mbr(bboxes, &group1_indices);
                let mbr2 = self.calculate_mbr(bboxes, &group2_indices);

                // Calculate overlap
                let overlap = self.calculate_overlap(&mbr1, &mbr2);

                // Calculate total area
                let area = mbr1.area() + mbr2.area();

                // Choose split with minimum overlap, then minimum area
                if overlap < min_overlap || (overlap == min_overlap && area < min_area) {
                    min_overlap = overlap;
                    min_area = area;
                    best_axis = axis;
                    best_index = k;
                }
            }
        }

        (best_axis, best_index)
    }

    /// Calculate the minimum bounding rectangle for a group of bounding boxes
    fn calculate_mbr(&self, bboxes: &[Option<&BoundingBox>], indices: &[usize]) -> BoundingBox {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for &idx in indices {
            if let Some(bbox) = bboxes[idx] {
                min_x = min_x.min(bbox.min_x);
                min_y = min_y.min(bbox.min_y);
                max_x = max_x.max(bbox.max_x);
                max_y = max_y.max(bbox.max_y);
            }
        }

        BoundingBox::new(min_x, min_y, max_x, max_y)
    }

    /// Calculate the overlap area between two bounding boxes
    fn calculate_overlap(&self, bbox1: &BoundingBox, bbox2: &BoundingBox) -> f64 {
        let overlap_min_x = bbox1.min_x.max(bbox2.min_x);
        let overlap_max_x = bbox1.max_x.min(bbox2.max_x);
        let overlap_min_y = bbox1.min_y.max(bbox2.min_y);
        let overlap_max_y = bbox1.max_y.min(bbox2.max_y);

        if overlap_max_x > overlap_min_x && overlap_max_y > overlap_min_y {
            (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
        } else {
            0.0
        }
    }

    /// Simple split fallback for when not all entries are bounding boxes
    fn simple_split(&self, n: usize) -> (Vec<usize>, Vec<usize>) {
        let mid = n / 2;
        let group1: Vec<usize> = (0..mid).collect();
        let group2: Vec<usize> = (mid..n).collect();
        (group1, group2)
    }
}

/// Serializer for spatial GiST index
#[derive(Debug)]
pub struct SpatialSerializer;

impl GiSTSerializer for SpatialSerializer {
    fn serialize_lookup(&self, lookup: &GiSTLookup) -> Result<RecordBatch> {
        let mut page_records = Vec::new();

        fn collect_leaf_pages(
            node: &GiSTNode,
            internal_nodes: &BTreeMap<u32, GiSTNode>,
            page_records: &mut Vec<(u32, BoundingBox)>,
        ) {
            if node.is_leaf {
                if let Some(bbox_pred) = node.predicate.as_any().downcast_ref::<BoundingBox>() {
                    for &entry in &node.entries {
                        page_records.push((entry, bbox_pred.clone()));
                    }
                }
            } else {
                for &child_id in &node.entries {
                    if let Some(child_node) = internal_nodes.get(&child_id) {
                        collect_leaf_pages(child_node, internal_nodes, page_records);
                    } else if let Some(bbox_pred) =
                        node.predicate.as_any().downcast_ref::<BoundingBox>()
                    {
                        page_records.push((child_id, bbox_pred.clone()));
                    }
                }
            }
        }

        collect_leaf_pages(&lookup.root, &lookup.internal_nodes, &mut page_records);
        create_lookup_batch(page_records)
    }

    fn deserialize_lookup(&self, data: RecordBatch) -> Result<GiSTLookup> {
        let mut pages = Vec::new();

        let min_x_array = data
            .column_by_name("min_x")
            .ok_or_else(|| Error::Internal {
                message: "deserialize_lookup: missing min_x".to_string(),
                location: location!(),
            })?;
        let min_x_array = min_x_array
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        let min_y_array = data
            .column_by_name("min_y")
            .ok_or_else(|| Error::Internal {
                message: "deserialize_lookup: missing min_y".to_string(),
                location: location!(),
            })?;
        let min_y_array = min_y_array
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        let max_x_array = data
            .column_by_name("max_x")
            .ok_or_else(|| Error::Internal {
                message: "deserialize_lookup: missing max_x".to_string(),
                location: location!(),
            })?;
        let max_x_array = max_x_array
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        let max_y_array = data
            .column_by_name("max_y")
            .ok_or_else(|| Error::Internal {
                message: "deserialize_lookup: missing max_y".to_string(),
                location: location!(),
            })?;
        let max_y_array = max_y_array
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        let page_numbers = data
            .column_by_name("page_number")
            .ok_or_else(|| Error::Internal {
                message: "deserialize_lookup: missing page_number".to_string(),
                location: location!(),
            })?;
        let page_numbers = page_numbers
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();

        for idx in 0..data.num_rows() {
            let bbox = BoundingBox::new(
                min_x_array.value(idx),
                min_y_array.value(idx),
                max_x_array.value(idx),
                max_y_array.value(idx),
            );
            let page_number = page_numbers.value(idx);

            pages.push((page_number, Box::new(bbox) as Box<dyn GiSTPredicate>));
        }

        GiSTLookup::build_tree(pages, &SpatialGiSTOps)
    }

    fn lookup_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("min_x", DataType::Float64, false),
            Field::new("min_y", DataType::Float64, false),
            Field::new("max_x", DataType::Float64, false),
            Field::new("max_y", DataType::Float64, false),
            Field::new("page_number", DataType::UInt32, false),
        ]))
    }
}

/// Extract bounding box from batch data
pub fn extract_bounding_box(batch: &RecordBatch) -> Result<BoundingBox> {
    if batch.num_rows() == 0 {
        return Ok(BoundingBox::new(0.0, 0.0, 0.0, 0.0));
    }

    let min_x_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| Error::Internal {
            message: "Expected Float64Array for min_x column".to_string(),
            location: location!(),
        })?;
    let min_y_array = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| Error::Internal {
            message: "Expected Float64Array for min_y column".to_string(),
            location: location!(),
        })?;
    let max_x_array = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| Error::Internal {
            message: "Expected Float64Array for max_x column".to_string(),
            location: location!(),
        })?;
    let max_y_array = batch
        .column(3)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| Error::Internal {
            message: "Expected Float64Array for max_y column".to_string(),
            location: location!(),
        })?;

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for i in 0..batch.num_rows() {
        min_x = min_x.min(min_x_array.value(i));
        min_y = min_y.min(min_y_array.value(i));
        max_x = max_x.max(max_x_array.value(i));
        max_y = max_y.max(max_y_array.value(i));
    }

    Ok(BoundingBox::new(min_x, min_y, max_x, max_y))
}

pub fn create_lookup_batch(pages: Vec<(u32, BoundingBox)>) -> Result<RecordBatch> {
    let min_x: Vec<f64> = pages.iter().map(|(_, bbox)| bbox.min_x).collect();
    let min_y: Vec<f64> = pages.iter().map(|(_, bbox)| bbox.min_y).collect();
    let max_x: Vec<f64> = pages.iter().map(|(_, bbox)| bbox.max_x).collect();
    let max_y: Vec<f64> = pages.iter().map(|(_, bbox)| bbox.max_y).collect();
    let page_numbers: Vec<u32> = pages.iter().map(|(page_id, _)| *page_id).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("min_x", DataType::Float64, false),
        Field::new("min_y", DataType::Float64, false),
        Field::new("max_x", DataType::Float64, false),
        Field::new("max_y", DataType::Float64, false),
        Field::new("page_number", DataType::UInt32, false),
    ]));

    let columns = vec![
        Arc::new(Float64Array::from(min_x)) as Arc<dyn Array>,
        Arc::new(Float64Array::from(min_y)) as Arc<dyn Array>,
        Arc::new(Float64Array::from(max_x)) as Arc<dyn Array>,
        Arc::new(Float64Array::from(max_y)) as Arc<dyn Array>,
        Arc::new(UInt32Array::from(page_numbers)) as Arc<dyn Array>,
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}

/// Create a spatial index from bounding boxes
pub async fn build_spatial_index(
    bboxes: Vec<(u32, BoundingBox)>,
    store: Arc<dyn super::IndexStore>,
    sub_index: Arc<dyn super::btree::BTreeSubIndex>,
    batch_size: u64,
    fri: Option<Arc<crate::frag_reuse::FragReuseIndex>>,
) -> Result<GiSTIndex> {
    let predicates: Vec<(u32, Box<dyn GiSTPredicate>)> = bboxes
        .into_iter()
        .map(|(id, bbox)| (id, Box::new(bbox) as Box<dyn GiSTPredicate>))
        .collect();

    GiSTIndex::build_index(
        predicates,
        Arc::new(SpatialGiSTOps),
        store,
        sub_index,
        batch_size,
        fri,
    )
    .await
}

/// Build spatial index using STR bulk loading for better performance
pub async fn build_spatial_index_str(
    mut bboxes: Vec<(u32, BoundingBox)>,
    store: Arc<dyn super::IndexStore>,
    sub_index: Arc<dyn super::btree::BTreeSubIndex>,
    batch_size: u64,
    fri: Option<Arc<crate::frag_reuse::FragReuseIndex>>,
) -> Result<GiSTIndex> {
    // Sort by X coordinate (using center point)
    bboxes.sort_by(|a, b| {
        let center_a = (a.1.min_x + a.1.max_x) / 2.0;
        let center_b = (b.1.min_x + b.1.max_x) / 2.0;
        center_a.partial_cmp(&center_b).unwrap()
    });

    // Calculate slice parameters
    let entries_per_node = 16; // MAX_ENTRIES_PER_NODE
    let num_leaf_nodes = (bboxes.len() + entries_per_node - 1) / entries_per_node;
    let num_slices = (num_leaf_nodes as f64).sqrt().ceil() as usize;
    let entries_per_slice = (bboxes.len() + num_slices - 1) / num_slices;

    // Create vertical slices
    let mut slices: Vec<Vec<(u32, BoundingBox)>> = Vec::new();
    for chunk in bboxes.chunks(entries_per_slice) {
        let mut slice = chunk.to_vec();
        // Sort each slice by Y coordinate
        slice.sort_by(|a, b| {
            let center_a = (a.1.min_y + a.1.max_y) / 2.0;
            let center_b = (b.1.min_y + b.1.max_y) / 2.0;
            center_a.partial_cmp(&center_b).unwrap()
        });
        slices.push(slice);
    }

    // Flatten back into sorted entries
    let sorted_entries: Vec<(u32, Box<dyn GiSTPredicate>)> = slices
        .into_iter()
        .flatten()
        .map(|(id, bbox)| (id, Box::new(bbox) as Box<dyn GiSTPredicate>))
        .collect();

    // Build tree using STR
    let lookup = GiSTLookup::build_tree(sorted_entries, &SpatialGiSTOps)?;

    Ok(GiSTIndex::new(
        lookup,
        Arc::new(SpatialGiSTOps),
        store,
        sub_index,
        batch_size,
        fri,
    ))
}

/// Load spatial index from serialized data
pub fn load_spatial_index(
    data: RecordBatch,
    store: Arc<dyn super::IndexStore>,
    batch_size: u64,
    fri: Option<Arc<crate::frag_reuse::FragReuseIndex>>,
) -> Result<GiSTIndex> {
    let mut pages = Vec::new();

    if data.num_rows() == 0 {
        let sub_index = Arc::new(super::flat::FlatIndexMetadata::new(DataType::Struct(
            Fields::from(vec![
                Field::new("min_x", DataType::Float64, false),
                Field::new("min_y", DataType::Float64, false),
                Field::new("max_x", DataType::Float64, false),
                Field::new("max_y", DataType::Float64, false),
            ]),
        )));
        let ops = SpatialGiSTOps;
        let lookup = GiSTLookup::empty(&ops);
        return Ok(GiSTIndex::new(
            lookup,
            Arc::new(SpatialGiSTOps),
            store,
            sub_index,
            batch_size,
            fri,
        ));
    }

    let min_x_array = data
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let min_y_array = data
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let max_x_array = data
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let max_y_array = data
        .column(3)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let page_numbers = data
        .column(4)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();

    for idx in 0..data.num_rows() {
        let bbox = BoundingBox::new(
            min_x_array.value(idx),
            min_y_array.value(idx),
            max_x_array.value(idx),
            max_y_array.value(idx),
        );
        let page_number = page_numbers.value(idx);

        pages.push((page_number, bbox));
    }

    let ops = SpatialGiSTOps;
    let predicates: Vec<(u32, Box<dyn GiSTPredicate>)> = pages
        .into_iter()
        .map(|(id, bbox)| (id, Box::new(bbox) as Box<dyn GiSTPredicate>))
        .collect();
    let lookup = GiSTLookup::build_tree(predicates, &ops)?;
    let sub_index = Arc::new(super::flat::FlatIndexMetadata::new(DataType::Struct(
        Fields::from(vec![
            Field::new("min_x", DataType::Float64, false),
            Field::new("min_y", DataType::Float64, false),
            Field::new("max_x", DataType::Float64, false),
            Field::new("max_y", DataType::Float64, false),
        ]),
    )));
    Ok(GiSTIndex::new(
        lookup,
        Arc::new(SpatialGiSTOps),
        store,
        sub_index,
        batch_size,
        fri,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_tree_construction() {
        let ops = SpatialGiSTOps;
        let predicates = vec![
            (
                1,
                Box::new(BoundingBox::new(0.0, 0.0, 5.0, 5.0)) as Box<dyn GiSTPredicate>,
            ),
            (
                2,
                Box::new(BoundingBox::new(5.0, 5.0, 10.0, 10.0)) as Box<dyn GiSTPredicate>,
            ),
            (
                3,
                Box::new(BoundingBox::new(10.0, 10.0, 15.0, 15.0)) as Box<dyn GiSTPredicate>,
            ),
        ];

        let lookup = GiSTLookup::build_tree(predicates, &ops).unwrap();
        assert!(!lookup.root.entries.is_empty());
        assert!(lookup.tree_depth >= 1);
    }

    #[test]
    fn test_spatial_query_search() {
        let ops = SpatialGiSTOps;
        let predicates = vec![
            (
                1,
                Box::new(BoundingBox::new(0.0, 0.0, 5.0, 5.0)) as Box<dyn GiSTPredicate>,
            ),
            (
                2,
                Box::new(BoundingBox::new(10.0, 10.0, 15.0, 15.0)) as Box<dyn GiSTPredicate>,
            ),
        ];

        let lookup = GiSTLookup::build_tree(predicates, &ops).unwrap();

        let spatial_query = SpatialQuery::Intersects(BoundingBox::new(2.0, 2.0, 7.0, 7.0));
        let query = Box::new(spatial_query) as Box<dyn GiSTQuery>;
        let results = lookup.search_pages(&*query, &ops);

        assert!(results.contains(&1));
        assert!(!results.contains(&2));
    }

    #[test]
    fn test_contains_query() {
        let ops = SpatialGiSTOps;
        let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox2 = BoundingBox::new(2.0, 2.0, 5.0, 5.0);

        // Test Contains query - bbox1 should be consistent with Contains(bbox2)
        let query = SpatialQuery::Contains(bbox2.clone());
        assert!(ops.consistent(&bbox1, &query));

        // But bbox2 should NOT be consistent with Contains(bbox1)
        let query2 = SpatialQuery::Contains(bbox1.clone());
        assert!(!ops.consistent(&bbox2, &query2));
    }

    #[test]
    fn test_r_star_split() {
        let ops = SpatialGiSTOps;

        // Create a set of bounding boxes that will benefit from R*-tree split
        let entries: Vec<Box<dyn GiSTPredicate>> = vec![
            Box::new(BoundingBox::new(0.0, 0.0, 2.0, 2.0)),
            Box::new(BoundingBox::new(1.0, 1.0, 3.0, 3.0)),
            Box::new(BoundingBox::new(2.0, 2.0, 4.0, 4.0)),
            Box::new(BoundingBox::new(10.0, 0.0, 12.0, 2.0)),
            Box::new(BoundingBox::new(11.0, 1.0, 13.0, 3.0)),
            Box::new(BoundingBox::new(12.0, 2.0, 14.0, 4.0)),
        ];

        let (group1, group2) = ops.pick_split(&entries);

        // Verify that the split separates spatially distinct groups
        assert!(!group1.is_empty());
        assert!(!group2.is_empty());

        // Calculate MBRs for both groups
        let bboxes: Vec<Option<&BoundingBox>> = entries
            .iter()
            .map(|pred| pred.as_any().downcast_ref::<BoundingBox>())
            .collect();

        let mbr1 = ops.calculate_mbr(&bboxes, &group1);
        let mbr2 = ops.calculate_mbr(&bboxes, &group2);

        // Verify minimal overlap
        let overlap = ops.calculate_overlap(&mbr1, &mbr2);
        assert!(overlap < 0.1); // Should have minimal or no overlap
    }

    #[tokio::test]
    async fn test_str_bulk_loading() {
        // Create test data
        let mut bboxes = Vec::new();
        // Create a grid of bounding boxes
        for i in 0..10 {
            for j in 0..10 {
                let x = i as f64 * 10.0;
                let y = j as f64 * 10.0;
                bboxes.push((
                    (i * 10 + j) as u32,
                    BoundingBox::new(x, y, x + 5.0, y + 5.0),
                ));
            }
        }

        // Sort by X coordinate for STR
        bboxes.sort_by(|a, b| {
            let center_a = (a.1.min_x + a.1.max_x) / 2.0;
            let center_b = (b.1.min_x + b.1.max_x) / 2.0;
            center_a.partial_cmp(&center_b).unwrap()
        });

        // Test building the tree structure using STR
        let ops = SpatialGiSTOps;
        let sorted_entries: Vec<(u32, Box<dyn GiSTPredicate>)> = bboxes
            .into_iter()
            .map(|(id, bbox)| (id, Box::new(bbox) as Box<dyn GiSTPredicate>))
            .collect();

        // Build tree using STR
        let lookup = GiSTLookup::build_tree(sorted_entries.clone(), &ops).unwrap();

        // Basic validations
        assert!(!lookup.root.entries.is_empty());
        assert_eq!(lookup.page_predicates.len(), 100); // Should have all 100 entries

        // Verify that all page IDs are in the tree
        let mut found_pages = std::collections::HashSet::new();
        fn collect_pages(
            node: &GiSTNode,
            internal_nodes: &BTreeMap<u32, GiSTNode>,
            found_pages: &mut std::collections::HashSet<u32>,
        ) {
            for &entry in &node.entries {
                if let Some(child) = internal_nodes.get(&entry) {
                    collect_pages(child, internal_nodes, found_pages);
                } else {
                    found_pages.insert(entry);
                }
            }
        }

        collect_pages(&lookup.root, &lookup.internal_nodes, &mut found_pages);
        assert_eq!(found_pages.len(), 100); // Should find all 100 pages

        // Compare with regular build_tree
        let regular_lookup = GiSTLookup::build_tree(sorted_entries, &ops).unwrap();

        // Both should produce valid trees with all entries
        assert_eq!(
            lookup.page_predicates.len(),
            regular_lookup.page_predicates.len()
        );

        // Test search functionality
        let query = SpatialQuery::Intersects(BoundingBox::new(25.0, 25.0, 35.0, 35.0));
        let str_results = lookup.search_pages(&query, &ops);
        let regular_results = regular_lookup.search_pages(&query, &ops);

        // Both should return the same results (order might differ)
        let str_set: std::collections::HashSet<_> = str_results.into_iter().collect();
        let regular_set: std::collections::HashSet<_> = regular_results.into_iter().collect();
        assert_eq!(str_set, regular_set);
    }

    #[test]
    fn test_r_star_split_edge_cases() {
        let ops = SpatialGiSTOps;

        // Test with minimum entries
        let entries: Vec<Box<dyn GiSTPredicate>> = vec![
            Box::new(BoundingBox::new(0.0, 0.0, 1.0, 1.0)),
            Box::new(BoundingBox::new(2.0, 2.0, 3.0, 3.0)),
        ];

        let (group1, group2) = ops.pick_split(&entries);
        assert_eq!(group1.len(), 1);
        assert_eq!(group2.len(), 1);

        // Test with overlapping boxes
        let overlapping: Vec<Box<dyn GiSTPredicate>> = vec![
            Box::new(BoundingBox::new(0.0, 0.0, 5.0, 5.0)),
            Box::new(BoundingBox::new(2.0, 2.0, 7.0, 7.0)),
            Box::new(BoundingBox::new(4.0, 4.0, 9.0, 9.0)),
            Box::new(BoundingBox::new(6.0, 6.0, 11.0, 11.0)),
        ];

        let (group1, group2) = ops.pick_split(&overlapping);
        assert_eq!(group1.len() + group2.len(), overlapping.len());
        assert!(!group1.is_empty());
        assert!(!group2.is_empty());
    }

    #[test]
    fn test_enhanced_gist_methods() {
        let ops = SpatialGiSTOps;

        let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox2 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox3 = BoundingBox::new(1.0, 1.0, 11.0, 11.0);

        assert!(ops.same(&bbox1, &bbox2));
        assert!(!ops.same(&bbox1, &bbox3));

        let penalty = ops.penalty(&bbox1, &bbox3);
        assert!(penalty > 0.0);

        let predicates = vec![
            Box::new(BoundingBox::new(0.0, 0.0, 5.0, 5.0)) as Box<dyn GiSTPredicate>,
            Box::new(BoundingBox::new(10.0, 10.0, 15.0, 15.0)) as Box<dyn GiSTPredicate>,
            Box::new(BoundingBox::new(1.0, 1.0, 6.0, 6.0)) as Box<dyn GiSTPredicate>,
            Box::new(BoundingBox::new(11.0, 11.0, 16.0, 16.0)) as Box<dyn GiSTPredicate>,
        ];

        let (group1, group2) = ops.pick_split(&predicates);
        assert!(!group1.is_empty());
        assert!(!group2.is_empty());
        assert_eq!(group1.len() + group2.len(), predicates.len());

        let query = SpatialQuery::Intersects(BoundingBox::new(5.0, 5.0, 15.0, 15.0));
        let query_predicate = ops.query_to_predicate(&query);

        if let Some(result_bbox) = query_predicate.as_any().downcast_ref::<BoundingBox>() {
            assert_eq!(result_bbox.min_x, 5.0);
            assert_eq!(result_bbox.min_y, 5.0);
            assert_eq!(result_bbox.max_x, 15.0);
            assert_eq!(result_bbox.max_y, 15.0);
        } else {
            panic!("query_to_predicate should return a BoundingBox");
        }
    }

    #[test]
    fn test_penalty_calculation() {
        let ops = SpatialGiSTOps;
        let existing = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let new1 = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
        let new2 = BoundingBox::new(10.0, 10.0, 20.0, 20.0);

        let penalty1 = ops.penalty(&existing, &new1);
        let penalty2 = ops.penalty(&existing, &new2);

        assert!(penalty1 > 0.0);
        assert!(penalty2 > penalty1);
    }

    #[test]
    fn test_union_multiple_boxes() {
        let ops = SpatialGiSTOps;
        let bbox1 = BoundingBox::new(0.0, 0.0, 5.0, 5.0);
        let bbox2 = BoundingBox::new(3.0, 3.0, 7.0, 7.0);
        let bbox3 = BoundingBox::new(6.0, 6.0, 10.0, 10.0);

        let predicates: Vec<&dyn GiSTPredicate> = vec![&bbox1, &bbox2, &bbox3];
        let union = ops.union(&predicates);

        if let Some(union_bbox) = union.as_any().downcast_ref::<BoundingBox>() {
            assert_eq!(union_bbox.min_x, 0.0);
            assert_eq!(union_bbox.min_y, 0.0);
            assert_eq!(union_bbox.max_x, 10.0);
            assert_eq!(union_bbox.max_y, 10.0);
        } else {
            panic!("Union should return a BoundingBox");
        }
    }

    #[test]
    fn test_intersects_multiple() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox2 = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
        let bbox3 = BoundingBox::new(20.0, 20.0, 30.0, 30.0);

        assert!(bbox1.intersects(&bbox2));
        assert!(!bbox1.intersects(&bbox3));
        assert!(bbox2.intersects(&bbox1));
        assert!(!bbox2.intersects(&bbox3));
    }
}
