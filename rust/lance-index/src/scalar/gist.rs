// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
    collections::{BTreeMap, HashMap},
    fmt,
    sync::Arc,
};

use super::{
    btree::BTreeSubIndex, flat::FlatIndexMetadata, AnyQuery, IndexStore, MetricsCollector,
    ScalarIndex, SearchResult,
};
use crate::frag_reuse::FragReuseIndex;
use crate::{Index, IndexType};
use arrow_array::{Array, Float64Array, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Fields, Schema};
use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_expr::Expr;
use deepsize::{Context, DeepSizeOf};
use futures::StreamExt;
use lance_core::{
    utils::{
        mask::RowIdTreeMap,
        tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS},
    },
    Error, Result,
};
use moka::sync::Cache;
use roaring::RoaringBitmap;
use serde::Serialize;
use snafu::location;
use tracing::info;

const GIST_LOOKUP_NAME: &str = "gist_page_lookup.lance";
const GIST_PAGES_NAME: &str = "gist_page_data.lance";
pub const DEFAULT_GIST_BATCH_SIZE: u64 = 4096;
const BATCH_SIZE_META_KEY: &str = "batch_size";
const MAX_ENTRIES_PER_NODE: usize = 16;

lazy_static::lazy_static! {
    static ref CACHE_SIZE: u64 = std::env::var("LANCE_SPATIAL_CACHE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512 * 1024 * 1024);
}

trait GiSTPredicate: Send + Sync + std::fmt::Debug + DeepSizeOf {
    fn clone_box(&self) -> Box<dyn GiSTPredicate>;
    fn as_any(&self) -> &dyn Any;
    fn eq_predicate(&self, other: &dyn GiSTPredicate) -> bool;
}

impl Clone for Box<dyn GiSTPredicate> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl PartialEq for Box<dyn GiSTPredicate> {
    fn eq(&self, other: &Self) -> bool {
        self.eq_predicate(other.as_ref())
    }
}

trait GiSTQuery: Send + Sync + DeepSizeOf + AnyQuery {
    fn as_any_query(&self) -> &dyn AnyQuery;
    fn clone_box(&self) -> Box<dyn GiSTQuery>;
}

impl Clone for Box<dyn GiSTQuery> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

trait GiSTOperations: Send + Sync + std::fmt::Debug + DeepSizeOf {
    fn consistent(&self, predicate: &dyn GiSTPredicate, query: &dyn GiSTQuery) -> bool;

    fn union(&self, predicates: &[&dyn GiSTPredicate]) -> Box<dyn GiSTPredicate>;

    fn same(&self, predicate: &dyn GiSTPredicate, other: &dyn GiSTPredicate) -> bool;

    fn penalty(&self, existing: &dyn GiSTPredicate, new: &dyn GiSTPredicate) -> f64;

    fn pick_split(&self, entries: &[Box<dyn GiSTPredicate>]) -> (Vec<usize>, Vec<usize>);

    fn query_to_predicate(&self, query: &dyn GiSTQuery) -> Box<dyn GiSTPredicate>;
}

#[derive(Debug, DeepSizeOf, PartialEq, Clone)]
#[repr(C, align(32))]
struct BoundingBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
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
    fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    fn intersects(&self, other: &Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    fn contains(&self, other: &Self) -> bool {
        self.min_x <= other.min_x
            && self.max_x >= other.max_x
            && self.min_y <= other.min_y
            && self.max_y >= other.max_y
    }

    fn area(&self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }
}

impl GiSTPredicate for BoundingBox {
    fn clone_box(&self) -> Box<dyn GiSTPredicate> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn eq_predicate(&self, other: &dyn GiSTPredicate) -> bool {
        if let Some(other_bbox) = other.as_any().downcast_ref::<BoundingBox>() {
            self == other_bbox
        } else {
            false
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
enum SpatialQuery {
    Intersects(BoundingBox),
    Contains(BoundingBox),
}

impl DeepSizeOf for SpatialQuery {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        std::mem::size_of::<BoundingBox>()
    }
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

    fn to_expr(&self, _col: String) -> Expr {
        todo!()
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

    fn clone_box(&self) -> Box<dyn GiSTQuery> {
        Box::new(self.clone())
    }
}

#[derive(Debug)]
struct SpatialGiSTOps;

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
                    bbox_pred.min_x <= query_bbox.min_x
                        && bbox_pred.max_x >= query_bbox.max_x
                        && bbox_pred.min_y <= query_bbox.min_y
                        && bbox_pred.max_y >= query_bbox.max_y
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

        let bbox_data: Vec<(usize, f64, f64, f64, f64)> = entries
            .iter()
            .enumerate()
            .filter_map(|(i, pred)| {
                pred.as_any()
                    .downcast_ref::<BoundingBox>()
                    .map(|bbox| (i, bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y))
            })
            .collect();

        if bbox_data.len() <= 2 {
            return (vec![0], if bbox_data.len() > 1 { vec![1] } else { vec![] });
        }

        let (seed1, seed2) = self.find_optimal_seeds(&bbox_data);
        self.distribute_entries(entries, seed1, seed2)
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
}

impl SpatialGiSTOps {
    fn find_optimal_seeds(&self, bbox_data: &[(usize, f64, f64, f64, f64)]) -> (usize, usize) {
        let mut max_separation = f64::NEG_INFINITY;
        let mut seed1 = 0;
        let mut seed2 = 1;

        for i in 0..bbox_data.len() {
            let (_, min_x1, min_y1, max_x1, max_y1) = unsafe { *bbox_data.get_unchecked(i) };
            let area1 = (max_x1 - min_x1) * (max_y1 - min_y1);

            for j in (i + 1)..bbox_data.len() {
                let (_, min_x2, min_y2, max_x2, max_y2) = unsafe { *bbox_data.get_unchecked(j) };
                let area2 = (max_x2 - min_x2) * (max_y2 - min_y2);

                let union_area = (max_x1.max(max_x2) - min_x1.min(min_x2))
                    * (max_y1.max(max_y2) - min_y1.min(min_y2));
                let separation = union_area - area1 - area2;

                if separation > max_separation {
                    max_separation = separation;
                    seed1 = bbox_data[i].0;
                    seed2 = bbox_data[j].0;
                }
            }
        }
        (seed1, seed2)
    }

    fn distribute_entries(
        &self,
        entries: &[Box<dyn GiSTPredicate>],
        seed1: usize,
        seed2: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut group1 = Vec::with_capacity(entries.len() / 2 + 1);
        let mut group2 = Vec::with_capacity(entries.len() / 2 + 1);

        group1.push(seed1);
        group2.push(seed2);

        let mut union1 = entries[seed1].clone();
        let mut union2 = entries[seed2].clone();

        for i in 0..entries.len() {
            if i == seed1 || i == seed2 {
                continue;
            }

            let penalty1 = self.penalty(union1.as_ref(), entries[i].as_ref());
            let penalty2 = self.penalty(union2.as_ref(), entries[i].as_ref());

            if penalty1 <= penalty2 && group1.len() < entries.len() - 1 {
                group1.push(i);
                union1 = self.union(&[union1.as_ref(), entries[i].as_ref()]);
            } else {
                group2.push(i);
                union2 = self.union(&[union2.as_ref(), entries[i].as_ref()]);
            }
        }

        (group1, group2)
    }
}

#[derive(Debug)]
struct GiSTNode {
    predicate: Box<dyn GiSTPredicate>,
    is_leaf: bool,
    entries: Vec<u32>,
}

#[derive(Debug)]
struct GiSTLookup {
    root: GiSTNode,
    internal_nodes: BTreeMap<u32, GiSTNode>,
    page_predicates: BTreeMap<u32, Box<dyn GiSTPredicate>>,
    max_entries_per_node: usize,
    tree_depth: u8,
}

impl GiSTLookup {
    fn new(
        root: GiSTNode,
        internal_nodes: BTreeMap<u32, GiSTNode>,
        page_predicates: BTreeMap<u32, Box<dyn GiSTPredicate>>,
        max_entries_per_node: usize,
        tree_depth: u8,
    ) -> Self {
        Self {
            root,
            internal_nodes,
            page_predicates,
            max_entries_per_node,
            tree_depth,
        }
    }

    fn empty() -> Self {
        let root = GiSTNode {
            predicate: Box::new(BoundingBox::new(0.0, 0.0, 0.0, 0.0)),
            is_leaf: true,
            entries: vec![],
        };
        Self::new(
            root,
            BTreeMap::new(),
            BTreeMap::new(),
            MAX_ENTRIES_PER_NODE,
            1,
        )
    }

    fn search_pages(&self, query: &dyn GiSTQuery, ops: &dyn GiSTOperations) -> Vec<u32> {
        let mut result = Vec::new();
        let mut to_visit = Vec::with_capacity(self.tree_depth as usize * MAX_ENTRIES_PER_NODE);

        // 使用 query_to_predicate 进行查询优化
        let query_predicate = ops.query_to_predicate(query);

        to_visit.push((&self.root, 0u8));

        while let Some((node, depth)) = to_visit.pop() {
            if !ops.consistent(&*node.predicate, query) {
                continue;
            }

            if node.is_leaf {
                result.reserve(node.entries.len());
                for &page_id in &node.entries {
                    if let Some(page_predicate) = self.page_predicates.get(&page_id) {
                        // 使用双重检查：先用 consistent，再用 query_predicate 进行精确匹配
                        if ops.consistent(&**page_predicate, query) {
                            // 进一步使用 query_predicate 进行精确过滤
                            if self.predicate_matches_query(&**page_predicate, &*query_predicate, ops) {
                                result.push(page_id);
                            }
                        }
                    }
                }
            } else {
                to_visit.reserve(node.entries.len());
                for &child_id in &node.entries {
                    if let Some(child_node) = self.internal_nodes.get(&child_id) {
                        to_visit.push((child_node, depth + 1));
                    } else {
                        result.push(child_id);
                    }
                }
            }
        }

        result
    }

    // 使用 query_predicate 进行精确匹配
    fn predicate_matches_query(
        &self,
        page_predicate: &dyn GiSTPredicate,
        query_predicate: &dyn GiSTPredicate,
        ops: &dyn GiSTOperations,
    ) -> bool {
        // 如果查询谓词和页面谓词相同，直接匹配
        if ops.same(page_predicate, query_predicate) {
            return true;
        }

        // 检查查询谓词是否与页面谓词有重叠
        // 这里可以根据具体的谓词类型进行更精确的匹配
        match (
            page_predicate.as_any().downcast_ref::<BoundingBox>(),
            query_predicate.as_any().downcast_ref::<BoundingBox>(),
        ) {
            (Some(page_bbox), Some(query_bbox)) => {
                // 检查边界框是否相交
                page_bbox.intersects(query_bbox)
            }
            _ => true, // 如果无法精确匹配，保守地返回 true
        }
    }

    // 动态插入新的谓词到现有树中
    fn insert_predicate(
        &mut self,
        page_id: u32,
        predicate: Box<dyn GiSTPredicate>,
        ops: &dyn GiSTOperations,
    ) -> Result<()> {
        // 检查是否已存在相同的谓词
        if let Some(existing_predicate) = self.page_predicates.get(&page_id) {
            if ops.same(&**existing_predicate, &*predicate) {
                return Ok(()); // 已存在相同谓词，无需插入
            }
        }

        // 存储页面谓词
        self.page_predicates.insert(page_id, predicate.clone());

        // 如果是空树，直接设置为根节点
        if self.root.entries.is_empty() {
            self.root.predicate = predicate;
            self.root.entries.push(page_id);
            return Ok(());
        }

        // 寻找最佳插入位置
        let best_node_id = self.find_best_insertion_node(&*predicate, ops)?;
        
        if let Some(node_id) = best_node_id {
            if let Some(node) = self.internal_nodes.get_mut(&node_id) {
                node.entries.push(page_id);
                // 更新节点的谓词为所有子项的联合
                let child_predicates: Vec<&dyn GiSTPredicate> = node.entries.iter()
                    .filter_map(|&id| self.page_predicates.get(&id).map(|p| &**p))
                    .collect();
                node.predicate = ops.union(&child_predicates);
            }
        } else {
            // 插入到根节点
            self.root.entries.push(page_id);
            let root_child_predicates: Vec<&dyn GiSTPredicate> = self.root.entries.iter()
                .filter_map(|&id| self.page_predicates.get(&id).map(|p| &**p))
                .collect();
            self.root.predicate = ops.union(&root_child_predicates);
        }

        Ok(())
    }

    // 使用 penalty() 方法寻找最佳插入位置
    fn find_best_insertion_node(
        &self,
        predicate: &dyn GiSTPredicate,
        ops: &dyn GiSTOperations,
    ) -> Result<Option<u32>> {
        let mut best_penalty = f64::INFINITY;
        let mut best_node_id = None;

        // 检查根节点
        let root_penalty = ops.penalty(&*self.root.predicate, predicate);
        if root_penalty < best_penalty {
            best_penalty = root_penalty;
            best_node_id = None; // None 表示根节点
        }

        // 检查所有内部节点
        for (&node_id, node) in &self.internal_nodes {
            let penalty = ops.penalty(&*node.predicate, predicate);
            if penalty < best_penalty {
                best_penalty = penalty;
                best_node_id = Some(node_id);
            }
        }

        Ok(best_node_id)
    }

    fn build_tree(
        leaf_pages: Vec<(u32, Box<dyn GiSTPredicate>)>,
        ops: &dyn GiSTOperations,
    ) -> Result<Self> {
        if leaf_pages.is_empty() {
            return Ok(Self::empty());
        }

        if leaf_pages.len() == 1 {
            let (page_id, predicate) = leaf_pages.into_iter().next().unwrap();
            let mut page_predicates = BTreeMap::new();
            page_predicates.insert(page_id, predicate.clone());

            let root = GiSTNode {
                predicate,
                is_leaf: true,
                entries: vec![page_id],
            };

            return Ok(Self::new(
                root,
                BTreeMap::new(),
                page_predicates,
                MAX_ENTRIES_PER_NODE,
                1,
            ));
        }

        // 使用 same() 方法进行去重优化
        let mut page_predicates: BTreeMap<u32, Box<dyn GiSTPredicate>> = BTreeMap::new();
        let mut predicate_cache: Vec<Box<dyn GiSTPredicate>> = Vec::new();
        
        for (page_id, predicate) in leaf_pages.iter().cloned() {
            // 检查是否已有相同的谓词
            let mut found_same = false;
            for cached_predicate in &predicate_cache {
                if ops.same(&**cached_predicate, &*predicate) {
                    // 复用现有的谓词实例
                    page_predicates.insert(page_id, cached_predicate.clone());
                    found_same = true;
                    break;
                }
            }
            
            if !found_same {
                // 添加新的谓词到缓存
                predicate_cache.push(predicate.clone());
                page_predicates.insert(page_id, predicate);
            }
        }
        let mut internal_nodes = BTreeMap::new();
        let mut next_node_id = leaf_pages.len() as u32;
        let mut current_level: Vec<(u32, Box<dyn GiSTPredicate>)> = leaf_pages;
        let mut tree_depth = 1u8;

        let estimated_nodes = current_level.len().div_ceil(MAX_ENTRIES_PER_NODE);
        let mut next_level = Vec::with_capacity(estimated_nodes);

        while current_level.len() > MAX_ENTRIES_PER_NODE {
            next_level.clear();

            // 使用智能分裂而不是简单分块
            let mut remaining_entries = current_level.clone();
            
            while !remaining_entries.is_empty() {
                let chunk_size = std::cmp::min(MAX_ENTRIES_PER_NODE * 2, remaining_entries.len());
                let chunk_entries: Vec<(u32, Box<dyn GiSTPredicate>)> = 
                    remaining_entries.drain(..chunk_size).collect();
                
                if chunk_entries.len() <= MAX_ENTRIES_PER_NODE {
                    // 直接创建节点
                    let predicates: Vec<&dyn GiSTPredicate> =
                        chunk_entries.iter().map(|(_, pred)| &**pred).collect();
                    let union_predicate = ops.union(&predicates);
                    let entries: Vec<u32> = chunk_entries.iter().map(|(page_id, _)| *page_id).collect();

                    let node = GiSTNode {
                        predicate: union_predicate.clone(),
                        is_leaf: false,
                        entries,
                    };

                    let node_id = next_node_id;
                    next_node_id += 1;
                    internal_nodes.insert(node_id, node);
                    next_level.push((node_id, union_predicate));
                } else {
                    // 使用 pick_split 进行智能分裂
                    let predicates: Vec<Box<dyn GiSTPredicate>> = 
                        chunk_entries.iter().map(|(_, pred)| pred.clone()).collect();
                    let (group1_indices, group2_indices) = ops.pick_split(&predicates);
                    
                    // 创建第一个组
                    if !group1_indices.is_empty() {
                        let group1_entries: Vec<u32> = group1_indices.iter()
                            .map(|&i| chunk_entries[i].0).collect();
                        let group1_preds: Vec<&dyn GiSTPredicate> = group1_indices.iter()
                            .map(|&i| &*chunk_entries[i].1).collect();
                        let union1 = ops.union(&group1_preds);
                        
                        let node1 = GiSTNode {
                            predicate: union1.clone(),
                            is_leaf: false,
                            entries: group1_entries,
                        };
                        
                        let node_id1 = next_node_id;
                        next_node_id += 1;
                        internal_nodes.insert(node_id1, node1);
                        next_level.push((node_id1, union1));
                    }
                    
                    // 创建第二个组
                    if !group2_indices.is_empty() {
                        let group2_entries: Vec<u32> = group2_indices.iter()
                            .map(|&i| chunk_entries[i].0).collect();
                        let group2_preds: Vec<&dyn GiSTPredicate> = group2_indices.iter()
                            .map(|&i| &*chunk_entries[i].1).collect();
                        let union2 = ops.union(&group2_preds);
                        
                        let node2 = GiSTNode {
                            predicate: union2.clone(),
                            is_leaf: false,
                            entries: group2_entries,
                        };
                        
                        let node_id2 = next_node_id;
                        next_node_id += 1;
                        internal_nodes.insert(node_id2, node2);
                        next_level.push((node_id2, union2));
                    }
                }
            }

            current_level = next_level.clone();
            tree_depth += 1;
        }

        let root = if current_level.len() == 1 && internal_nodes.contains_key(&current_level[0].0) {
            internal_nodes.remove(&current_level[0].0).unwrap()
        } else {
            let predicates: Vec<&dyn GiSTPredicate> =
                current_level.iter().map(|(_, pred)| &**pred).collect();
            let union_predicate = ops.union(&predicates);

            GiSTNode {
                predicate: union_predicate,
                is_leaf: current_level.len() <= MAX_ENTRIES_PER_NODE,
                entries: current_level.iter().map(|(page_id, _)| *page_id).collect(),
            }
        };

        Ok(Self::new(
            root,
            internal_nodes,
            page_predicates,
            MAX_ENTRIES_PER_NODE,
            tree_depth,
        ))
    }
}

impl DeepSizeOf for GiSTLookup {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.root.deep_size_of_children(context)
            + self.internal_nodes.deep_size_of_children(context)
            + self.page_predicates.deep_size_of_children(context)
    }
}

impl DeepSizeOf for GiSTNode {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.predicate.deep_size_of_children(context) + self.entries.deep_size_of_children(context)
    }
}

impl DeepSizeOf for SpatialGiSTOps {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

#[derive(Debug)]
struct GiSTCache(Cache<u32, Arc<dyn ScalarIndex>>);

impl DeepSizeOf for GiSTCache {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        self.0.iter().map(|(_, v)| v.deep_size_of()).sum()
    }
}

#[derive(Debug, DeepSizeOf)]
pub struct GiSTIndex {
    lookup: Arc<GiSTLookup>,
    operations: Arc<dyn GiSTOperations>,
    cache: Arc<GiSTCache>,
    store: Arc<dyn IndexStore>,
    sub_index: Arc<dyn BTreeSubIndex>,
    batch_size: u64,
    fri: Option<Arc<FragReuseIndex>>,
}

pub type SpatialIndex = GiSTIndex;

impl GiSTIndex {
    fn new(
        lookup: GiSTLookup,
        operations: Arc<dyn GiSTOperations>,
        store: Arc<dyn IndexStore>,
        sub_index: Arc<dyn BTreeSubIndex>,
        batch_size: u64,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Self {
        let lookup = Arc::new(lookup);
        let cache = Arc::new(GiSTCache(
            Cache::builder()
                .max_capacity(*CACHE_SIZE)
                .weigher(|_, v: &Arc<dyn ScalarIndex>| v.deep_size_of() as u32)
                .build(),
        ));

        Self {
            lookup,
            operations,
            cache,
            store,
            sub_index,
            batch_size,
            fri,
        }
    }

    fn new_spatial(
        lookup: GiSTLookup,
        store: Arc<dyn IndexStore>,
        sub_index: Arc<dyn BTreeSubIndex>,
        batch_size: u64,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Self {
        Self::new(
            lookup,
            Arc::new(SpatialGiSTOps),
            store,
            sub_index,
            batch_size,
            fri,
        )
    }

    async fn lookup_page(
        &self,
        page_number: u32,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        if let Some(index) = self.cache.0.get(&page_number) {
            return Ok(index);
        }

        metrics.record_part_load();
        info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_SCALAR_PART, index_type="gist", part_id=page_number);

        let index_reader = self.store.open_index_file(GIST_PAGES_NAME).await?;
        let mut serialized_page = index_reader
            .read_record_batch(page_number as u64, self.batch_size)
            .await?;

        if let Some(fri_ref) = self.fri.as_ref() {
            serialized_page = fri_ref.remap_row_ids_record_batch(serialized_page, 1)?;
        }

        let subindex = self.sub_index.load_subindex(serialized_page).await?;
        self.cache.0.insert(page_number, subindex.clone());

        Ok(subindex)
    }

    async fn search_page(
        &self,
        query: &dyn AnyQuery,
        page_number: u32,
        metrics: &dyn MetricsCollector,
    ) -> Result<RowIdTreeMap> {
        let subindex = self.lookup_page(page_number, metrics).await?;

        match subindex.search(query, metrics).await? {
            SearchResult::Exact(map) => Ok(map),
            _ => Err(Error::Internal {
                message: "GiST sub-indices need to return exact results".to_string(),
                location: location!(),
            }),
        }
    }

    async fn load_existing_pages(&self) -> Result<Vec<(u32, BoundingBox)>> {
        let mut pages = Vec::new();

        for (&page_id, predicate) in &self.lookup.page_predicates {
            if let Some(bbox_pred) = predicate.as_bbox() {
                pages.push((
                    page_id,
                    BoundingBox::new(
                        bbox_pred.min_x,
                        bbox_pred.min_y,
                        bbox_pred.max_x,
                        bbox_pred.max_y,
                    ),
                ));
            }
        }

        pages.sort_by_key(|(page_id, _)| *page_id);
        Ok(pages)
    }

    async fn search(
        &self,
        query: &dyn GiSTQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let page_numbers = self.lookup.search_pages(query, self.operations.as_ref());

        if page_numbers.is_empty() {
            return Ok(SearchResult::Exact(RowIdTreeMap::default()));
        }

        let query_ref: &dyn AnyQuery = query.as_any_query();

        if page_numbers.len() == 1 {
            return Ok(SearchResult::Exact(
                self.search_page(query_ref, page_numbers[0], metrics)
                    .await?,
            ));
        }

        match page_numbers.len() {
            2..=4 => {
                let mut overall_row_ids = RowIdTreeMap::default();
                for page_number in page_numbers {
                    let page_result = self.search_page(query_ref, page_number, metrics).await?;
                    overall_row_ids |= page_result;
                }
                Ok(SearchResult::Exact(overall_row_ids))
            }
            5..=16 => {
                self.search_batch_optimized(query_ref, &page_numbers, metrics)
                    .await
            }
            _ => {
                self.search_large_batch(query_ref, &page_numbers, metrics)
                    .await
            }
        }
    }

    async fn search_batch_optimized(
        &self,
        query: &dyn AnyQuery,
        page_numbers: &[u32],
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let mut overall_row_ids = RowIdTreeMap::default();

        for chunk in page_numbers.chunks(4) {
            let mut tasks = Vec::with_capacity(chunk.len());

            for &page_number in chunk {
                tasks.push(self.search_page(query, page_number, metrics));
            }

            for task in tasks {
                let page_result = task.await?;
                overall_row_ids |= page_result;
            }
        }

        Ok(SearchResult::Exact(overall_row_ids))
    }

    async fn search_large_batch(
        &self,
        query: &dyn AnyQuery,
        page_numbers: &[u32],
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let mut overall_row_ids = RowIdTreeMap::default();

        const BATCH_SIZE: usize = 8;

        for chunk in page_numbers.chunks(BATCH_SIZE) {
            for &page_number in chunk {
                let page_result = self.search_page(query, page_number, metrics).await?;
                overall_row_ids |= page_result;

                if page_numbers.len() > 100 && page_number % 20 == 0 {
                    tokio::task::yield_now().await;
                }
            }
        }

        Ok(SearchResult::Exact(overall_row_ids))
    }

    async fn build_index(
        predicates: Vec<(u32, Box<dyn GiSTPredicate>)>,
        ops: &dyn GiSTOperations,
        store: Arc<dyn IndexStore>,
        sub_index: Arc<dyn BTreeSubIndex>,
        batch_size: u64,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let lookup = GiSTLookup::build_tree(predicates, ops)?;

        Ok(Self::new(
            lookup,
            Arc::new(SpatialGiSTOps),
            store,
            sub_index,
            batch_size,
            fri,
        ))
    }

    async fn build_from_bboxes(
        bboxes: Vec<(u32, BoundingBox)>,
        store: Arc<dyn IndexStore>,
        sub_index: Arc<dyn BTreeSubIndex>,
        batch_size: u64,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let predicates: Vec<(u32, Box<dyn GiSTPredicate>)> = bboxes
            .into_iter()
            .map(|(id, bbox)| (id, Box::new(bbox) as Box<dyn GiSTPredicate>))
            .collect();

        Self::build_index(
            predicates,
            &SpatialGiSTOps,
            store,
            sub_index,
            batch_size,
            fri,
        )
        .await
    }

    // 公共方法：动态插入新的边界框
    async fn insert_bbox(
        &self,
        page_id: u32,
        bbox: BoundingBox,
    ) -> Result<()> {
        let predicate = Box::new(bbox) as Box<dyn GiSTPredicate>;
        
        // 获取可变引用来修改 lookup
        let mut lookup = Arc::try_unwrap(self.lookup.clone()).map_err(|_| Error::Internal {
            message: "Cannot get mutable reference to lookup".to_string(),
            location: location!(),
        })?;
        
        lookup.insert_predicate(page_id, predicate, self.operations.as_ref())?;
        
        // 重新包装为 Arc
        let updated_lookup = Arc::new(lookup);
        
        Ok(())
    }

    fn try_from_serialized(
        data: RecordBatch,
        store: Arc<dyn IndexStore>,
        batch_size: u64,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let mut pages = Vec::new();

        if data.num_rows() == 0 {
            let sub_index = Arc::new(FlatIndexMetadata::new(DataType::Struct(Fields::from(
                vec![
                    Field::new("min_x", DataType::Float64, false),
                    Field::new("min_y", DataType::Float64, false),
                    Field::new("max_x", DataType::Float64, false),
                    Field::new("max_y", DataType::Float64, false),
                ],
            ))));
            let lookup = GiSTLookup::empty();
            return Ok(Self::new(
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
        let sub_index = Arc::new(FlatIndexMetadata::new(DataType::Struct(Fields::from(
            vec![
                Field::new("min_x", DataType::Float64, false),
                Field::new("min_y", DataType::Float64, false),
                Field::new("max_x", DataType::Float64, false),
                Field::new("max_y", DataType::Float64, false),
            ],
        ))));
        Ok(Self::new(
            lookup,
            Arc::new(SpatialGiSTOps),
            store,
            sub_index,
            batch_size,
            fri,
        ))
    }
}

#[derive(Serialize)]
struct SpatialStatistics {
    num_pages: u32,
    total_area: f64,
    avg_page_area: f64,
    tree_depth: u8,
    max_entries_per_node: usize,
    total_nodes: u32,
    leaf_nodes: u32,
    internal_nodes: u32,
}

#[async_trait]
impl Index for SpatialIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "SpatialIndex is not a vector index".into(),
            location: location!(),
        })
    }

    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::GiST
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let leaf_nodes = if self.lookup.root.is_leaf { 1 } else { 0 };
        let internal_nodes =
            self.lookup.internal_nodes.len() as u32 + if self.lookup.root.is_leaf { 0 } else { 1 };
        let total_nodes = leaf_nodes + internal_nodes;

        let mut total_area = 0.0;
        let mut num_pages = if self.lookup.root.is_leaf {
            self.lookup.root.entries.len()
        } else {
            0
        };

        if let Some(bbox_pred) = self
            .lookup
            .root
            .predicate
            .as_any()
            .downcast_ref::<BoundingBox>()
        {
            total_area += bbox_pred.area();
        }

        for node in self.lookup.internal_nodes.values() {
            if let Some(bbox_pred) = node.predicate.as_any().downcast_ref::<BoundingBox>() {
                total_area += bbox_pred.area();
            }
            if node.is_leaf {
                num_pages += node.entries.len();
            }
        }

        let avg_area = if total_nodes > 0 {
            total_area / total_nodes as f64
        } else {
            0.0
        };

        serde_json::to_value(&SpatialStatistics {
            num_pages: num_pages as u32,
            total_area,
            avg_page_area: avg_area,
            tree_depth: self.lookup.tree_depth,
            max_entries_per_node: self.lookup.max_entries_per_node,
            total_nodes,
            leaf_nodes,
            internal_nodes,
        })
        .map_err(|err| err.into())
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = RoaringBitmap::default();

        let sub_index_reader = self.store.open_index_file(GIST_PAGES_NAME).await?;
        let num_batches = sub_index_reader.num_batches(self.batch_size).await;

        for batch_idx in 0..num_batches {
            let serialized = sub_index_reader
                .read_record_batch(batch_idx as u64, self.batch_size)
                .await?;
            let page = self.sub_index.load_subindex(serialized).await?;
            frag_ids |= page.calculate_included_frags().await?;
        }

        Ok(frag_ids)
    }
}

#[async_trait]
impl ScalarIndex for SpatialIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let spatial_query: Box<dyn GiSTQuery> =
            if let Some(spatial_query) = query.as_any().downcast_ref::<SpatialQuery>() {
                Box::new(spatial_query.clone())
            } else {
                return Err(Error::Internal {
                    message: "Unsupported query type for GiST index".to_string(),
                    location: location!(),
                });
            };

        self.search(spatial_query.as_ref(), metrics).await
    }

    fn can_answer_exact(&self, _: &dyn AnyQuery) -> bool {
        true
    }

    async fn load(
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Arc<Self>> {
        let lookup_file = store.open_index_file(GIST_LOOKUP_NAME).await?;
        let num_rows = lookup_file.num_rows();
        let serialized_lookup = lookup_file.read_range(0..num_rows, None).await?;

        let file_schema = lookup_file.schema();
        let batch_size = file_schema
            .metadata
            .get(BATCH_SIZE_META_KEY)
            .map(|bs| bs.parse().unwrap_or(DEFAULT_GIST_BATCH_SIZE))
            .unwrap_or(DEFAULT_GIST_BATCH_SIZE);

        Ok(Arc::new(Self::try_from_serialized(
            serialized_lookup,
            store,
            batch_size,
            fri,
        )?))
    }

    async fn remap(
        &self,
        mapping: &HashMap<u64, Option<u64>>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let mut sub_index_file = dest_store
            .new_index_file(GIST_PAGES_NAME, self.sub_index.schema().clone())
            .await?;

        let sub_index_reader = self.store.open_index_file(GIST_PAGES_NAME).await?;
        let num_batches = sub_index_reader.num_batches(self.batch_size).await;

        for batch_idx in 0..num_batches {
            let serialized = sub_index_reader
                .read_record_batch(batch_idx as u64, self.batch_size)
                .await?;
            let remapped = self.sub_index.remap_subindex(serialized, mapping).await?;
            sub_index_file.write_record_batch(remapped).await?;
        }

        sub_index_file.finish().await?;

        self.store
            .copy_index_file(GIST_LOOKUP_NAME, dest_store)
            .await
    }

    async fn update(
        &self,
        mut new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let existing_pages = self.load_existing_pages().await?;
        let mut next_page_id = existing_pages.len() as u32;

        let mut sub_index_file = dest_store
            .new_index_file(GIST_PAGES_NAME, self.sub_index.schema().clone())
            .await?;

        let existing_reader = self.store.open_index_file(GIST_PAGES_NAME).await?;
        let num_existing_batches = existing_reader.num_batches(self.batch_size).await;
        for batch_idx in 0..num_existing_batches {
            let existing_batch = existing_reader
                .read_record_batch(batch_idx as u64, self.batch_size)
                .await?;
            sub_index_file.write_record_batch(existing_batch).await?;
        }

        let mut all_pages = existing_pages;
        let mut has_new_data = false;

        while let Some(batch_result) = new_data.next().await {
            let batch = batch_result?;
            has_new_data = true;

            let bbox = extract_bounding_box(&batch)?;
            let serialized = self.sub_index.train(batch).await?;
            sub_index_file.write_record_batch(serialized).await?;

            all_pages.push((next_page_id, bbox));
            next_page_id += 1;
        }

        sub_index_file.finish().await?;

        if !has_new_data {
            self.store
                .copy_index_file(GIST_LOOKUP_NAME, dest_store)
                .await?;
            return Ok(());
        }

        let ops = SpatialGiSTOps;
        let predicates: Vec<(u32, Box<dyn GiSTPredicate>)> = all_pages
            .into_iter()
            .map(|(id, bbox)| (id, Box::new(bbox) as Box<dyn GiSTPredicate>))
            .collect();
        let lookup = GiSTLookup::build_tree(predicates, &ops)?;
        serialize_lookup(&lookup, dest_store, self.batch_size as u32).await?;

        Ok(())
    }
}

fn extract_bounding_box(batch: &RecordBatch) -> Result<BoundingBox> {
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

async fn serialize_lookup(
    lookup: &GiSTLookup,
    index_store: &dyn IndexStore,
    batch_size: u32,
) -> Result<()> {
    let mut page_records = Vec::new();

    fn collect_leaf_pages(
        node: &GiSTNode,
        internal_nodes: &BTreeMap<u32, GiSTNode>,
        page_records: &mut Vec<(u32, BoundingBox)>,
    ) {
        if node.is_leaf {
            if let Some(bbox_pred) = node.predicate.as_bbox() {
                for &entry in &node.entries {
                    page_records.push((entry, bbox_pred.clone()));
                }
            }
        } else {
            for &child_id in &node.entries {
                if let Some(child_node) = internal_nodes.get(&child_id) {
                    collect_leaf_pages(child_node, internal_nodes, page_records);
                } else if let Some(bbox_pred) = node.predicate.as_bbox() {
                    page_records.push((child_id, bbox_pred.clone()));
                }
            }
        }
    }

    collect_leaf_pages(&lookup.root, &lookup.internal_nodes, &mut page_records);

    let lookup_batch = create_lookup_batch(page_records)?;
    let mut file_schema = lookup_batch.schema().as_ref().clone();
    file_schema
        .metadata
        .insert(BATCH_SIZE_META_KEY.to_string(), batch_size.to_string());

    let mut lookup_file = index_store
        .new_index_file(GIST_LOOKUP_NAME, Arc::new(file_schema))
        .await?;
    lookup_file.write_record_batch(lookup_batch).await?;
    lookup_file.finish().await?;

    Ok(())
}

fn create_lookup_batch(pages: Vec<(u32, BoundingBox)>) -> Result<RecordBatch> {
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

trait PredicateDowncast {
    fn as_bbox(&self) -> Option<&BoundingBox>;
}

impl PredicateDowncast for dyn GiSTPredicate {
    fn as_bbox(&self) -> Option<&BoundingBox> {
        self.as_any().downcast_ref::<BoundingBox>()
    }
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
    fn test_optimized_performance() {
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

        let spatial_query = SpatialQuery::Intersects(BoundingBox::new(0.5, 0.5, 2.5, 2.5));
        let query = Box::new(spatial_query) as Box<dyn GiSTQuery>;

        let results = lookup.search_pages(&*query, &ops);

        assert!(
            !results.is_empty(),
            "Query should return some results. Tree has {} pages, query covers area 0.5-2.5",
            lookup.page_predicates.len()
        );
    }

    #[test]
    fn test_enhanced_gist_methods() {
        let ops = SpatialGiSTOps;
        
        // 测试 same() 方法
        let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox2 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox3 = BoundingBox::new(1.0, 1.0, 11.0, 11.0);
        
        assert!(ops.same(&bbox1, &bbox2)); // 相同的边界框
        assert!(!ops.same(&bbox1, &bbox3)); // 不同的边界框
        
        // 测试 penalty() 方法
        let penalty = ops.penalty(&bbox1, &bbox3);
        assert!(penalty > 0.0); // penalty 应该大于 0
        
        // 测试 pick_split() 方法
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
        
        // 测试 query_to_predicate() 方法
        let query = SpatialQuery::Intersects(BoundingBox::new(5.0, 5.0, 15.0, 15.0));
        let query_predicate = ops.query_to_predicate(&query);
        
        // 验证转换的谓词不为空
        if let Some(result_bbox) = query_predicate.as_any().downcast_ref::<BoundingBox>() {
            assert_eq!(result_bbox.min_x, 5.0);
            assert_eq!(result_bbox.min_y, 5.0);
            assert_eq!(result_bbox.max_x, 15.0);
            assert_eq!(result_bbox.max_y, 15.0);
        } else {
            panic!("query_to_predicate should return a BoundingBox");
        }
        
        println!("All enhanced GiST methods are working correctly!");
    }
}
