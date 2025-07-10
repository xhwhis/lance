// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    any::Any,
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use super::{
    btree::BTreeSubIndex, AnyQuery, IndexStore, MetricsCollector, ScalarIndex, SearchResult,
};
use crate::frag_reuse::FragReuseIndex;
use crate::scalar::spatial::SpatialQuery;
use crate::{Index, IndexType};
use arrow_array::{Array, RecordBatch};
use arrow_schema::Schema;
use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use deepsize::{Context, DeepSizeOf};
use futures::{FutureExt, StreamExt, TryStreamExt};
use lance_core::{
    utils::{
        mask::RowIdTreeMap,
        tokio::get_num_compute_intensive_cpus,
        tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS},
    },
    Error, Result,
};
use moka::sync::Cache;
use roaring::RoaringBitmap;
use snafu::location;
use tracing::info;

const GIST_LOOKUP_NAME: &str = "gist_page_lookup.lance";
const GIST_PAGES_NAME: &str = "gist_page_data.lance";
pub const DEFAULT_GIST_BATCH_SIZE: u64 = 4096;
const MAX_ENTRIES_PER_NODE: usize = 16;

lazy_static::lazy_static! {
    static ref CACHE_SIZE: u64 = std::env::var("LANCE_GIST_CACHE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512 * 1024 * 1024);
}

// We only need to open a file reader for pages if we need to load a page.  If all
// pages are cached we don't open it.  If we do open it we should only open it once.
#[derive(Clone)]
struct LazyIndexReader {
    index_reader: Arc<tokio::sync::Mutex<Option<Arc<dyn super::IndexReader>>>>,
    store: Arc<dyn IndexStore>,
}

impl LazyIndexReader {
    fn new(store: Arc<dyn IndexStore>) -> Self {
        Self {
            index_reader: Arc::new(tokio::sync::Mutex::new(None)),
            store,
        }
    }

    async fn get(&self) -> Result<Arc<dyn super::IndexReader>> {
        let mut reader = self.index_reader.lock().await;
        if reader.is_none() {
            let index_reader = self.store.open_index_file(GIST_PAGES_NAME).await?;
            *reader = Some(index_reader);
        }
        Ok(reader.as_ref().unwrap().clone())
    }
}

/// Trait for GiST predicates (e.g., bounding boxes, intervals, etc.)
pub trait GiSTPredicate: Send + Sync + std::fmt::Debug + DeepSizeOf {
    fn as_any(&self) -> &dyn Any;
    fn dyn_clone(&self) -> Box<dyn GiSTPredicate>;
}

impl Clone for Box<dyn GiSTPredicate> {
    fn clone(&self) -> Self {
        self.dyn_clone()
    }
}

/// Trait for serializing GiST predicates
pub trait GiSTSerializable: GiSTPredicate {
    /// Serialize the predicate to Arrow arrays
    fn serialize(&self) -> Result<Vec<Arc<dyn Array>>>;

    /// Deserialize from Arrow arrays at a specific row
    fn deserialize(arrays: &[Arc<dyn Array>], row: usize) -> Result<Box<dyn GiSTPredicate>>
    where
        Self: Sized;

    /// Get the schema for serialized predicates
    fn schema() -> Arc<Schema>
    where
        Self: Sized;
}

/// Trait for GiST queries
pub trait GiSTQuery: Send + Sync + DeepSizeOf + AnyQuery {
    fn as_any_query(&self) -> &dyn AnyQuery;
}

/// Trait for GiST operations - defines the behavior of a specific index type
pub trait GiSTOperations: Send + Sync + std::fmt::Debug + DeepSizeOf {
    /// Check if a predicate is consistent with a query
    fn consistent(&self, predicate: &dyn GiSTPredicate, query: &dyn GiSTQuery) -> bool;

    /// Compute the union of multiple predicates
    fn union(&self, predicates: &[&dyn GiSTPredicate]) -> Box<dyn GiSTPredicate>;

    /// Check if two predicates are the same
    fn same(&self, predicate: &dyn GiSTPredicate, other: &dyn GiSTPredicate) -> bool;

    /// Compute the penalty for inserting a new predicate into an existing one
    fn penalty(&self, existing: &dyn GiSTPredicate, new: &dyn GiSTPredicate) -> f64;

    /// Split entries into two groups for node splitting
    fn pick_split(&self, entries: &[Box<dyn GiSTPredicate>]) -> (Vec<usize>, Vec<usize>);

    /// Convert a query to a predicate (for tree traversal)
    fn query_to_predicate(&self, query: &dyn GiSTQuery) -> Box<dyn GiSTPredicate>;

    /// Create an empty predicate
    fn empty_predicate(&self) -> Box<dyn GiSTPredicate>;

    /// Get the serializer for this operation type
    fn serializer(&self) -> Box<dyn GiSTSerializer>;

    /// Convert a batch of data into pages, returning the page data and their predicates
    fn batch_to_pages(
        &self,
        batch: RecordBatch,
        next_page_id: u32,
    ) -> Result<(Vec<RecordBatch>, Vec<(u32, Box<dyn GiSTPredicate>)>)>;
}

/// Trait for serializing GiST data structures
pub trait GiSTSerializer: Send + Sync {
    /// Serialize a lookup structure
    fn serialize_lookup(&self, lookup: &GiSTLookup) -> Result<RecordBatch>;

    /// Deserialize a lookup structure
    fn deserialize_lookup(&self, data: RecordBatch) -> Result<GiSTLookup>;

    /// Get schema for serialized lookup
    fn lookup_schema(&self) -> Arc<Schema>;
}

#[derive(Debug, DeepSizeOf, Clone)]
pub struct GiSTNode {
    pub predicate: Box<dyn GiSTPredicate>,
    pub is_leaf: bool,
    pub entries: Vec<u32>,
}

#[derive(Debug, DeepSizeOf, Clone)]
pub struct GiSTLookup {
    pub root: GiSTNode,
    pub internal_nodes: BTreeMap<u32, GiSTNode>,
    pub page_predicates: BTreeMap<u32, Box<dyn GiSTPredicate>>,
    pub max_entries_per_node: usize,
    pub tree_depth: u8,
    next_node_id: u32,
}

impl GiSTLookup {
    pub fn new(
        root: GiSTNode,
        internal_nodes: BTreeMap<u32, GiSTNode>,
        page_predicates: BTreeMap<u32, Box<dyn GiSTPredicate>>,
        max_entries_per_node: usize,
        tree_depth: u8,
        next_node_id: u32,
    ) -> Self {
        Self {
            root,
            internal_nodes,
            page_predicates,
            max_entries_per_node,
            tree_depth,
            next_node_id,
        }
    }

    pub fn empty(ops: &dyn GiSTOperations) -> Self {
        let root = GiSTNode {
            predicate: ops.empty_predicate(),
            is_leaf: true,
            entries: vec![],
        };
        Self::new(
            root,
            BTreeMap::new(),
            BTreeMap::new(),
            MAX_ENTRIES_PER_NODE,
            1,
            1,
        )
    }

    pub fn search_pages(&self, query: &dyn GiSTQuery, ops: &dyn GiSTOperations) -> Vec<u32> {
        let mut result = Vec::new();
        let mut to_visit = Vec::with_capacity(self.tree_depth as usize * MAX_ENTRIES_PER_NODE);

        to_visit.push((&self.root, 0u8));

        while let Some((node, depth)) = to_visit.pop() {
            if !ops.consistent(&*node.predicate, query) {
                continue;
            }

            if node.is_leaf {
                result.reserve(node.entries.len());
                for &page_id in &node.entries {
                    if let Some(page_predicate) = self.page_predicates.get(&page_id) {
                        if ops.consistent(&**page_predicate, query) {
                            result.push(page_id);
                        }
                    }
                }
            } else {
                to_visit.reserve(node.entries.len());
                for &child_id in &node.entries {
                    if let Some(child_node) = self.internal_nodes.get(&child_id) {
                        to_visit.push((child_node, depth + 1));
                    } else {
                        if let Some(page_predicate) = self.page_predicates.get(&child_id) {
                            if ops.consistent(&**page_predicate, query) {
                                result.push(child_id);
                            }
                        }
                    }
                }
            }
        }

        result
    }

    pub fn insert_predicate(
        &self,
        page_id: u32,
        predicate: Box<dyn GiSTPredicate>,
        ops: &dyn GiSTOperations,
    ) -> Result<Self> {
        if let Some(existing_predicate) = self.page_predicates.get(&page_id) {
            if ops.same(&**existing_predicate, &*predicate) {
                return Ok(self.clone());
            }
        }

        let mut state = CoWState::new(self);
        state.page_predicates.insert(page_id, predicate.clone());

        let mut root = self.root.clone();
        let mut tree_depth = self.tree_depth;

        if root.entries.is_empty() {
            root.predicate = predicate;
            root.entries.push(page_id);
            return Ok(state.into_lookup(root, tree_depth));
        }

        if let Some((new_node_id, new_node_predicate)) =
            Self::insert(&mut state, None, page_id, predicate, ops)?
        {
            // Root was split, create a new root
            let old_root = root;
            let old_root_id = state.next_node_id;
            state.next_node_id += 1;

            let old_root_predicate = old_root.predicate.clone();
            state.internal_nodes.insert(old_root_id, old_root);

            let new_root_predicate = ops.union(&[&*old_root_predicate, &*new_node_predicate]);
            root = GiSTNode {
                predicate: new_root_predicate,
                is_leaf: false,
                entries: vec![old_root_id, new_node_id],
            };
            tree_depth += 1;
        } else {
            // Update root predicate if it changed
            root = state.get_node_or_root(None).clone();
        }

        Ok(state.into_lookup(root, tree_depth))
    }

    // Inserts an entry into a node. If the node splits, the new node (id and predicate)
    // is returned to be inserted into the parent.
    fn insert(
        state: &mut CoWState,
        node_id: Option<u32>,
        entry_id: u32,
        entry_predicate: Box<dyn GiSTPredicate>,
        ops: &dyn GiSTOperations,
    ) -> Result<Option<(u32, Box<dyn GiSTPredicate>)>> {
        let node_is_leaf = state.get_node_or_root(node_id).is_leaf;
        let original_predicate = state.get_node_or_root(node_id).predicate.clone();

        // Update predicate
        let new_predicate = ops.union(&[&*original_predicate, &*entry_predicate]);
        state.get_mut_node_or_root(node_id).predicate = new_predicate;

        if node_is_leaf {
            state.get_mut_node_or_root(node_id).entries.push(entry_id);
        } else {
            // Choose subtree and recurse
            let mut best_child_id: Option<u32> = None;
            let mut min_penalty = f64::INFINITY;

            for &child_id in &state.get_node_or_root(node_id).entries {
                let child_predicate = if let Some(node) = state.get_node(child_id) {
                    &*node.predicate
                } else if let Some(pred) = state.page_predicates.get(&child_id) {
                    &**pred
                } else {
                    return Err(Error::Internal {
                        message: format!("GiST inconsistent: child {} not found.", child_id),
                        location: location!(),
                    });
                };

                let penalty = ops.penalty(child_predicate, &*entry_predicate);
                if penalty < min_penalty {
                    min_penalty = penalty;
                    best_child_id = Some(child_id);
                }
            }
            let chosen_id = best_child_id.unwrap();

            if let Some((new_child_id, new_child_predicate)) =
                Self::insert(state, Some(chosen_id), entry_id, entry_predicate, ops)?
            {
                let node = state.get_mut_node_or_root(node_id);
                node.entries.push(new_child_id);
                node.predicate = ops.union(&[&*node.predicate, &*new_child_predicate]);
            }
        }

        // Handle overflow and split
        if state.get_node_or_root(node_id).entries.len() > state.original.max_entries_per_node {
            let all_entry_predicates: Vec<Box<dyn GiSTPredicate>> = {
                let node = state.get_node_or_root(node_id);
                node.entries
                    .iter()
                    .map(|&id| {
                        if let Some(child_node) = state.get_node(id) {
                            child_node.predicate.clone()
                        } else {
                            state.page_predicates.get(&id).unwrap().clone()
                        }
                    })
                    .collect()
            };

            let (group1_indices, group2_indices) = ops.pick_split(&all_entry_predicates);

            let new_node_id = state.next_node_id;
            state.next_node_id += 1;

            let group2_entries = {
                let node = state.get_mut_node_or_root(node_id);
                let group2_entries: Vec<u32> = group2_indices
                    .iter()
                    .map(|&i| node.entries[i])
                    .collect();

                let group1_entries: Vec<u32> = group1_indices
                    .iter()
                    .map(|&i| node.entries[i])
                    .collect();
                node.entries = group1_entries;
                group2_entries
            };

            let group2_preds: Vec<&dyn GiSTPredicate> = group2_indices
                .iter()
                .map(|&i| &*all_entry_predicates[i])
                .collect();
            let union2 = ops.union(&group2_preds);

            let new_node = GiSTNode {
                predicate: union2.clone(),
                is_leaf: state.get_node_or_root(node_id).is_leaf,
                entries: group2_entries,
            };
            state.internal_nodes.insert(new_node_id, new_node);

            let group1_preds: Vec<&dyn GiSTPredicate> = group1_indices
                .iter()
                .map(|&i| &*all_entry_predicates[i])
                .collect();
            let union1 = ops.union(&group1_preds);

            state.get_mut_node_or_root(node_id).predicate = union1;

            return Ok(Some((new_node_id, union2)));
        }

        Ok(None)
    }

    pub fn build_tree(
        leaf_pages: Vec<(u32, Box<dyn GiSTPredicate>)>,
        ops: &dyn GiSTOperations,
    ) -> Result<Self> {
        if leaf_pages.is_empty() {
            return Ok(Self::empty(ops));
        }

        let num_pages = leaf_pages.len();
        if num_pages == 1 {
            let (page_id, predicate) = leaf_pages.into_iter().next().unwrap();
            let mut page_predicates = BTreeMap::new();
            page_predicates.insert(page_id, predicate.dyn_clone());

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
                num_pages as u32 + 1,
            ));
        }

        let mut page_predicates: BTreeMap<u32, Box<dyn GiSTPredicate>> = BTreeMap::new();
        let mut predicate_cache: Vec<Box<dyn GiSTPredicate>> = Vec::new();

        for (page_id, predicate) in leaf_pages.iter().cloned() {
            let mut found_same = false;
            for cached_predicate in &predicate_cache {
                if ops.same(&**cached_predicate, &*predicate) {
                    page_predicates.insert(page_id, cached_predicate.clone());
                    found_same = true;
                    break;
                }
            }

            if !found_same {
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

            let mut remaining_entries = current_level.clone();

            while !remaining_entries.is_empty() {
                let chunk_size = std::cmp::min(MAX_ENTRIES_PER_NODE * 2, remaining_entries.len());
                let chunk_entries: Vec<(u32, Box<dyn GiSTPredicate>)> =
                    remaining_entries.drain(..chunk_size).collect();

                if chunk_entries.len() <= MAX_ENTRIES_PER_NODE {
                    let predicates: Vec<&dyn GiSTPredicate> =
                        chunk_entries.iter().map(|(_, pred)| &**pred).collect();
                    let union_predicate = ops.union(&predicates);
                    let entries: Vec<u32> =
                        chunk_entries.iter().map(|(page_id, _)| *page_id).collect();

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
                    let predicates: Vec<Box<dyn GiSTPredicate>> =
                        chunk_entries.iter().map(|(_, pred)| pred.clone()).collect();
                    let (group1_indices, group2_indices) = ops.pick_split(&predicates);

                    if !group1_indices.is_empty() {
                        let group1_entries: Vec<u32> =
                            group1_indices.iter().map(|&i| chunk_entries[i].0).collect();
                        let group1_preds: Vec<&dyn GiSTPredicate> = group1_indices
                            .iter()
                            .map(|&i| &*chunk_entries[i].1)
                            .collect();
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

                    if !group2_indices.is_empty() {
                        let group2_entries: Vec<u32> =
                            group2_indices.iter().map(|&i| chunk_entries[i].0).collect();
                        let group2_preds: Vec<&dyn GiSTPredicate> = group2_indices
                            .iter()
                            .map(|&i| &*chunk_entries[i].1)
                            .collect();
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
            next_node_id,
        ))
    }
}

struct CoWState<'a> {
    original: &'a GiSTLookup,
    internal_nodes: BTreeMap<u32, GiSTNode>,
    page_predicates: BTreeMap<u32, Box<dyn GiSTPredicate>>,
    next_node_id: u32,
}

impl<'a> CoWState<'a> {
    fn new(original: &'a GiSTLookup) -> Self {
        Self {
            original,
            internal_nodes: BTreeMap::new(),
            page_predicates: original.page_predicates.clone(),
            next_node_id: original.next_node_id,
        }
    }

    fn get_node(&self, node_id: u32) -> Option<&GiSTNode> {
        self.internal_nodes
            .get(&node_id)
            .or_else(|| self.original.internal_nodes.get(&node_id))
    }

    fn get_node_or_root(&self, node_id: Option<u32>) -> &GiSTNode {
        if let Some(id) = node_id {
            self.get_node(id).unwrap()
        } else {
            self.internal_nodes.get(&0).unwrap_or(&self.original.root)
        }
    }

    fn get_mut_node_or_root(&mut self, node_id: Option<u32>) -> &mut GiSTNode {
        if let Some(id) = node_id {
            if !self.internal_nodes.contains_key(&id) {
                let original_node = self.original.internal_nodes.get(&id).unwrap().clone();
                self.internal_nodes.insert(id, original_node);
            }
            self.internal_nodes.get_mut(&id).unwrap()
        } else {
            if !self.internal_nodes.contains_key(&0) {
                self.internal_nodes.insert(0, self.original.root.clone());
            }
            self.internal_nodes.get_mut(&0).unwrap()
        }
    }

    fn into_lookup(self, root: GiSTNode, tree_depth: u8) -> GiSTLookup {
        let mut final_nodes = self.original.internal_nodes.clone();
        final_nodes.extend(self.internal_nodes);
        GiSTLookup::new(
            root,
            final_nodes,
            self.page_predicates,
            self.original.max_entries_per_node,
            tree_depth,
            self.next_node_id,
        )
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

impl GiSTIndex {
    pub fn new(
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

    async fn lookup_page(
        &self,
        page_number: u32,
        index_reader: LazyIndexReader,
        metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        if let Some(index) = self.cache.0.get(&page_number) {
            return Ok(index);
        }

        metrics.record_part_load();
        info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_SCALAR_PART, index_type="gist", part_id=page_number);

        let index_reader = index_reader.get().await?;
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
        index_reader: LazyIndexReader,
        metrics: &dyn MetricsCollector,
    ) -> Result<RowIdTreeMap> {
        let subindex = self.lookup_page(page_number, index_reader, metrics).await?;

        match subindex.search(query, metrics).await? {
            SearchResult::Exact(map) => Ok(map),
            _ => Err(Error::Internal {
                message: "GiST sub-indices need to return exact results".to_string(),
                location: location!(),
            }),
        }
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
        let lazy_index_reader = LazyIndexReader::new(self.store.clone());

        if page_numbers.len() == 1 {
            return Ok(SearchResult::Exact(
                self.search_page(query_ref, page_numbers[0], lazy_index_reader, metrics)
                    .await?,
            ));
        }

        match page_numbers.len() {
            2..=4 => {
                let mut overall_row_ids = RowIdTreeMap::default();
                for page_number in page_numbers {
                    let page_result = self
                        .search_page(query_ref, page_number, lazy_index_reader.clone(), metrics)
                        .await?;
                    overall_row_ids |= page_result;
                }
                Ok(SearchResult::Exact(overall_row_ids))
            }
            5..=16 => {
                self.search_batch_optimized(query_ref, &page_numbers, lazy_index_reader, metrics)
                    .await
            }
            _ => {
                self.search_large_batch(query_ref, &page_numbers, lazy_index_reader, metrics)
                    .await
            }
        }
    }

    async fn search_batch_optimized(
        &self,
        query: &dyn AnyQuery,
        page_numbers: &[u32],
        index_reader: LazyIndexReader,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let page_tasks = page_numbers
            .iter()
            .map(|&page_number| {
                self.search_page(query, page_number, index_reader.clone(), metrics)
                    .boxed()
            })
            .collect::<Vec<_>>();

        let row_ids = futures::stream::iter(page_tasks)
            .buffered(4)
            .try_collect::<RowIdTreeMap>()
            .await?;

        Ok(SearchResult::Exact(row_ids))
    }

    async fn search_large_batch(
        &self,
        query: &dyn AnyQuery,
        page_numbers: &[u32],
        index_reader: LazyIndexReader,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let page_tasks = page_numbers
            .iter()
            .map(|&page_number| {
                self.search_page(query, page_number, index_reader.clone(), metrics)
                    .boxed()
            })
            .collect::<Vec<_>>();

        let row_ids = futures::stream::iter(page_tasks)
            // Use compute intensive thread count for parallelism
            .buffered(get_num_compute_intensive_cpus())
            .try_collect::<RowIdTreeMap>()
            .await?;

        Ok(SearchResult::Exact(row_ids))
    }

    pub async fn build_index(
        predicates: Vec<(u32, Box<dyn GiSTPredicate>)>,
        operations: Arc<dyn GiSTOperations>,
        store: Arc<dyn IndexStore>,
        sub_index: Arc<dyn BTreeSubIndex>,
        batch_size: u64,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let lookup = GiSTLookup::build_tree(predicates, operations.as_ref())?;

        Ok(Self::new(
            lookup, operations, store, sub_index, batch_size, fri,
        ))
    }
}

#[async_trait]
impl Index for GiSTIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn as_vector_index(self: Arc<Self>) -> Result<Arc<dyn crate::vector::VectorIndex>> {
        Err(Error::NotSupported {
            source: "GiSTIndex is not a vector index".into(),
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

        let num_pages = if self.lookup.root.is_leaf {
            self.lookup.root.entries.len()
        } else {
            self.lookup.page_predicates.len()
        };

        let stats = serde_json::json!({
            "num_pages": num_pages as u32,
            "tree_depth": self.lookup.tree_depth,
            "max_entries_per_node": self.lookup.max_entries_per_node,
            "total_nodes": total_nodes,
            "leaf_nodes": leaf_nodes,
            "internal_nodes": internal_nodes,
        });

        Ok(stats)
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
impl ScalarIndex for GiSTIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        if let Some(spatial_query) = query.as_any().downcast_ref::<SpatialQuery>() {
            self.search(spatial_query, metrics).await
        } else {
            Err(Error::Internal {
                message: "Query type not supported by GiSTIndex".to_string(),
                location: location!(),
            })
        }
    }

    fn can_answer_exact(&self, _: &dyn AnyQuery) -> bool {
        true
    }

    async fn load(
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<FragReuseIndex>>,
    ) -> Result<Arc<Self>> {
        let lookup_reader = store.open_index_file(GIST_LOOKUP_NAME).await?;
        // The lookup file is a single batch file.
        let lookup_batch = lookup_reader
            .read_range(0..lookup_reader.num_rows(), None)
            .await?;

        // TODO: once we support more GiST variants, we need a way to
        // know which serializer and ops to use.
        let serializer = super::spatial::SpatialSerializer;
        let lookup = serializer.deserialize_lookup(lookup_batch)?;

        let sub_index = Arc::new(super::flat::FlatIndexMetadata::new(
            arrow_schema::DataType::Struct(arrow_schema::Fields::from(vec![
                arrow_schema::Field::new("min_x", arrow_schema::DataType::Float64, false),
                arrow_schema::Field::new("min_y", arrow_schema::DataType::Float64, false),
                arrow_schema::Field::new("max_x", arrow_schema::DataType::Float64, false),
                arrow_schema::Field::new("max_y", arrow_schema::DataType::Float64, false),
            ])),
        ));

        Ok(Arc::new(GiSTIndex::new(
            lookup,
            Arc::new(super::spatial::SpatialGiSTOps),
            store,
            sub_index,
            DEFAULT_GIST_BATCH_SIZE,
            fri,
        )))
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
        // Start with the existing lookup tree.
        let mut current_lookup = self.lookup.as_ref().clone();

        let mut sub_index_file = dest_store
            .new_index_file(GIST_PAGES_NAME, self.sub_index.schema().clone())
            .await?;

        // Copy existing pages
        let existing_reader = self.store.open_index_file(GIST_PAGES_NAME).await?;
        let num_existing_batches = existing_reader.num_batches(self.batch_size).await;
        for batch_idx in 0..num_existing_batches {
            let existing_batch = existing_reader
                .read_record_batch(batch_idx as u64, self.batch_size)
                .await?;
            sub_index_file.write_record_batch(existing_batch).await?;
        }

        let mut next_page_id = current_lookup.page_predicates.len() as u32;
        let mut has_new_data = false;

        while let Some(batch_result) = new_data.next().await {
            let batch = batch_result?;
            has_new_data = true;

            let (page_batches, page_predicates) =
                self.operations.batch_to_pages(batch, next_page_id)?;

            for page_batch in page_batches {
                sub_index_file.write_record_batch(page_batch).await?;
            }

            let num_new_pages = page_predicates.len() as u32;
            for (page_id, predicate) in page_predicates {
                // Perform CoW insertion, getting a new lookup object back.
                current_lookup = current_lookup.insert_predicate(
                    page_id,
                    predicate,
                    self.operations.as_ref(),
                )?;
            }

            next_page_id += num_new_pages;
        }

        sub_index_file.finish().await?;

        if !has_new_data {
            // No new data, just copy the lookup file
            self.store
                .copy_index_file(GIST_LOOKUP_NAME, dest_store)
                .await?;
            return Ok(());
        }

        // Write the new lookup file
        let serializer = self.operations.serializer();
        let lookup_batch = serializer.serialize_lookup(&current_lookup)?;
        let mut lookup_file = dest_store
            .new_index_file(GIST_LOOKUP_NAME, serializer.lookup_schema())
            .await?;
        lookup_file.write_record_batch(lookup_batch).await?;
        lookup_file.finish().await?;

        Ok(())
    }
}
