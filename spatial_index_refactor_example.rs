// 空间索引重构实施示例
// 此文件展示了如何将现有的 GiST 实现重构为更高效的空间索引

use std::collections::BTreeMap;
use std::sync::Arc;

// ============================================================================
// 1. 重构前的问题代码（简化版本）
// ============================================================================

// 原始的 trait object 方式 - 性能差，内存占用高
trait GiSTPredicate: Send + Sync + std::fmt::Debug {
    fn clone_box(&self) -> Box<dyn GiSTPredicate>;
    fn as_any(&self) -> &dyn std::any::Any;
}

trait GiSTOperations: Send + Sync {
    fn consistent(&self, predicate: &dyn GiSTPredicate, query: &dyn GiSTQuery) -> bool;
    fn union(&self, predicates: &[&dyn GiSTPredicate]) -> Box<dyn GiSTPredicate>;
    // 未使用的方法
    fn same(&self, predicate: &dyn GiSTPredicate, other: &dyn GiSTPredicate) -> bool;
    fn penalty(&self, existing: &dyn GiSTPredicate, new: &dyn GiSTPredicate) -> f64;
    fn pick_split(&self, entries: &[Box<dyn GiSTPredicate>]) -> (Vec<usize>, Vec<usize>);
    fn query_to_predicate(&self, query: &dyn GiSTQuery) -> Box<dyn GiSTPredicate>;
}

trait GiSTQuery: Send + Sync {
    fn clone_box(&self) -> Box<dyn GiSTQuery>;
}

struct GiSTNode {
    predicate: Box<dyn GiSTPredicate>,
    is_leaf: bool,
    entries: Vec<u32>,
}

// ============================================================================
// 2. 重构后的优化代码
// ============================================================================

// 2.1 直接使用具体类型，避免 trait object
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BoundingBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BoundingBox {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }

    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    #[inline]
    pub fn contains(&self, other: &Self) -> bool {
        self.min_x <= other.min_x
            && self.max_x >= other.max_x
            && self.min_y <= other.min_y
            && self.max_y >= other.max_y
    }

    #[inline]
    pub fn union_with(&self, other: &Self) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    #[inline]
    pub fn area(&self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }
}

// 2.2 使用枚举而非 trait object
#[derive(Debug, Clone, PartialEq)]
pub enum SpatialQuery {
    Intersects(BoundingBox),
    Contains(BoundingBox),
    Within(BoundingBox),
    Overlaps(BoundingBox),
}

impl SpatialQuery {
    #[inline]
    pub fn matches(&self, bbox: &BoundingBox) -> bool {
        match self {
            SpatialQuery::Intersects(query_bbox) => bbox.intersects(query_bbox),
            SpatialQuery::Contains(query_bbox) => bbox.contains(query_bbox),
            SpatialQuery::Within(query_bbox) => query_bbox.contains(bbox),
            SpatialQuery::Overlaps(query_bbox) => bbox.intersects(query_bbox) && !bbox.contains(query_bbox) && !query_bbox.contains(bbox),
        }
    }

    pub fn get_bbox(&self) -> &BoundingBox {
        match self {
            SpatialQuery::Intersects(bbox) |
            SpatialQuery::Contains(bbox) |
            SpatialQuery::Within(bbox) |
            SpatialQuery::Overlaps(bbox) => bbox,
        }
    }
}

// 2.3 简化的节点结构
#[derive(Debug, Clone)]
pub struct SpatialNode {
    pub bbox: BoundingBox,
    pub is_leaf: bool,
    pub entries: Vec<u32>,
}

impl SpatialNode {
    pub fn new(bbox: BoundingBox, is_leaf: bool, entries: Vec<u32>) -> Self {
        Self { bbox, is_leaf, entries }
    }

    pub fn new_leaf(bbox: BoundingBox, entries: Vec<u32>) -> Self {
        Self::new(bbox, true, entries)
    }

    pub fn new_internal(bbox: BoundingBox, entries: Vec<u32>) -> Self {
        Self::new(bbox, false, entries)
    }
}

// 2.4 优化的空间树结构
#[derive(Debug)]
pub struct SpatialTree {
    root: SpatialNode,
    internal_nodes: BTreeMap<u32, SpatialNode>,
    leaf_bboxes: BTreeMap<u32, BoundingBox>,
    config: SpatialTreeConfig,
}

#[derive(Debug, Clone)]
pub struct SpatialTreeConfig {
    pub max_entries_per_node: usize,
    pub tree_depth: u8,
}

impl Default for SpatialTreeConfig {
    fn default() -> Self {
        Self {
            max_entries_per_node: 16,
            tree_depth: 1,
        }
    }
}

impl SpatialTree {
    pub fn new(config: SpatialTreeConfig) -> Self {
        let root = SpatialNode::new_leaf(
            BoundingBox::new(0.0, 0.0, 0.0, 0.0),
            vec![]
        );
        
        Self {
            root,
            internal_nodes: BTreeMap::new(),
            leaf_bboxes: BTreeMap::new(),
            config,
        }
    }

    // 重构后的搜索方法 - 避免动态分发，使用迭代而非递归
    pub fn search_pages(&self, query: &SpatialQuery) -> Vec<u32> {
        let mut result = Vec::new();
        let mut to_visit = Vec::with_capacity(self.config.tree_depth as usize * self.config.max_entries_per_node);
        
        to_visit.push(&self.root);

        while let Some(node) = to_visit.pop() {
            if !query.matches(&node.bbox) {
                continue;
            }

            if node.is_leaf {
                // 批量处理叶子节点
                result.extend(
                    node.entries.iter()
                        .filter_map(|&page_id| {
                            self.leaf_bboxes.get(&page_id)
                                .filter(|bbox| query.matches(bbox))
                                .map(|_| page_id)
                        })
                );
            } else {
                // 预分配容量，避免重复分配
                to_visit.reserve(node.entries.len());
                for &child_id in &node.entries {
                    if let Some(child_node) = self.internal_nodes.get(&child_id) {
                        to_visit.push(child_node);
                    }
                }
            }
        }

        result
    }

    // 优化的树构建方法 - 移除未使用的复杂分裂逻辑
    pub fn build_from_pages(pages: Vec<(u32, BoundingBox)>) -> Self {
        let mut config = SpatialTreeConfig::default();
        
        if pages.is_empty() {
            return Self::new(config);
        }

        if pages.len() == 1 {
            let (page_id, bbox) = pages.into_iter().next().unwrap();
            let mut leaf_bboxes = BTreeMap::new();
            leaf_bboxes.insert(page_id, bbox);

            let root = SpatialNode::new_leaf(bbox, vec![page_id]);
            
            return Self {
                root,
                internal_nodes: BTreeMap::new(),
                leaf_bboxes,
                config,
            };
        }

        let leaf_bboxes: BTreeMap<u32, BoundingBox> = pages.iter().cloned().collect();
        let mut internal_nodes = BTreeMap::new();
        let mut next_node_id = pages.len() as u32;
        let mut current_level: Vec<(u32, BoundingBox)> = pages;
        let mut tree_depth = 1u8;

        // 简化的树构建 - 使用分块而非复杂的分裂算法
        while current_level.len() > config.max_entries_per_node {
            let mut next_level = Vec::new();

            for chunk in current_level.chunks(config.max_entries_per_node) {
                let union_bbox = chunk.iter()
                    .map(|(_, bbox)| bbox)
                    .fold(chunk[0].1, |acc, bbox| acc.union_with(bbox));

                let entries: Vec<u32> = chunk.iter().map(|(page_id, _)| *page_id).collect();
                let node = SpatialNode::new_internal(union_bbox, entries);

                let node_id = next_node_id;
                next_node_id += 1;
                internal_nodes.insert(node_id, node);
                next_level.push((node_id, union_bbox));
            }

            current_level = next_level;
            tree_depth += 1;
        }

        let root = if current_level.len() == 1 && internal_nodes.contains_key(&current_level[0].0) {
            internal_nodes.remove(&current_level[0].0).unwrap()
        } else {
            let union_bbox = current_level.iter()
                .map(|(_, bbox)| bbox)
                .fold(current_level[0].1, |acc, bbox| acc.union_with(bbox));

            SpatialNode::new(
                union_bbox,
                current_level.len() <= config.max_entries_per_node,
                current_level.iter().map(|(page_id, _)| *page_id).collect(),
            )
        };

        config.tree_depth = tree_depth;

        Self {
            root,
            internal_nodes,
            leaf_bboxes,
            config,
        }
    }

    // 统计信息方法
    pub fn statistics(&self) -> SpatialTreeStatistics {
        SpatialTreeStatistics {
            total_pages: self.leaf_bboxes.len() as u32,
            total_nodes: self.internal_nodes.len() as u32 + 1, // +1 for root
            tree_depth: self.config.tree_depth,
            max_entries_per_node: self.config.max_entries_per_node,
            avg_entries_per_node: if self.internal_nodes.is_empty() {
                self.root.entries.len() as f64
            } else {
                self.internal_nodes.values()
                    .map(|node| node.entries.len())
                    .sum::<usize>() as f64 / self.internal_nodes.len() as f64
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpatialTreeStatistics {
    pub total_pages: u32,
    pub total_nodes: u32,
    pub tree_depth: u8,
    pub max_entries_per_node: usize,
    pub avg_entries_per_node: f64,
}

// 2.5 进一步优化：紧凑的边界框表示
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CompactBoundingBox {
    bounds: [f32; 4], // [min_x, min_y, max_x, max_y]
}

impl CompactBoundingBox {
    pub fn new(min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Self {
        Self {
            bounds: [min_x, min_y, max_x, max_y],
        }
    }

    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        self.bounds[0] <= other.bounds[2] && // min_x <= other.max_x
        self.bounds[2] >= other.bounds[0] && // max_x >= other.min_x
        self.bounds[1] <= other.bounds[3] && // min_y <= other.max_y
        self.bounds[3] >= other.bounds[1]    // max_y >= other.min_y
    }

    #[inline]
    pub fn area(&self) -> f32 {
        (self.bounds[2] - self.bounds[0]) * (self.bounds[3] - self.bounds[1])
    }

    pub fn from_bbox(bbox: &BoundingBox) -> Self {
        Self::new(
            bbox.min_x as f32,
            bbox.min_y as f32,
            bbox.max_x as f32,
            bbox.max_y as f32,
        )
    }
}

// ============================================================================
// 3. 性能测试和基准
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_bbox_operations() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox2 = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
        
        assert!(bbox1.intersects(&bbox2));
        assert!(!bbox1.contains(&bbox2));
        
        let union = bbox1.union_with(&bbox2);
        assert_eq!(union, BoundingBox::new(0.0, 0.0, 15.0, 15.0));
        
        assert_eq!(bbox1.area(), 100.0);
    }

    #[test]
    fn test_spatial_query() {
        let bbox = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let query = SpatialQuery::Intersects(BoundingBox::new(5.0, 5.0, 15.0, 15.0));
        
        assert!(query.matches(&bbox));
        
        let query2 = SpatialQuery::Contains(BoundingBox::new(20.0, 20.0, 25.0, 25.0));
        assert!(!query2.matches(&bbox));
    }

    #[test]
    fn test_tree_construction() {
        let pages = vec![
            (1, BoundingBox::new(0.0, 0.0, 10.0, 10.0)),
            (2, BoundingBox::new(10.0, 0.0, 20.0, 10.0)),
            (3, BoundingBox::new(0.0, 10.0, 10.0, 20.0)),
            (4, BoundingBox::new(10.0, 10.0, 20.0, 20.0)),
        ];
        
        let tree = SpatialTree::build_from_pages(pages);
        let stats = tree.statistics();
        
        assert_eq!(stats.total_pages, 4);
        assert!(stats.tree_depth >= 1);
    }

    #[test]
    fn test_search_performance() {
        // 创建大量测试数据
        let mut pages = Vec::new();
        for i in 0..10000 {
            let x = (i % 100) as f64 * 10.0;
            let y = (i / 100) as f64 * 10.0;
            pages.push((i as u32, BoundingBox::new(x, y, x + 10.0, y + 10.0)));
        }
        
        let tree = SpatialTree::build_from_pages(pages);
        let query = SpatialQuery::Intersects(BoundingBox::new(450.0, 450.0, 550.0, 550.0));
        
        let start = Instant::now();
        let results = tree.search_pages(&query);
        let duration = start.elapsed();
        
        println!("Search took: {:?}", duration);
        println!("Found {} results", results.len());
        
        // 确保搜索结果正确
        assert!(!results.is_empty());
        assert!(results.len() < 10000); // 应该只返回匹配的结果
    }

    #[test]
    fn test_compact_bbox_memory() {
        // 验证紧凑边界框确实使用更少内存
        assert_eq!(std::mem::size_of::<BoundingBox>(), 32);
        assert_eq!(std::mem::size_of::<CompactBoundingBox>(), 16);
        
        let bbox = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let compact = CompactBoundingBox::from_bbox(&bbox);
        let other_compact = CompactBoundingBox::new(5.0, 5.0, 15.0, 15.0);
        
        assert!(compact.intersects(&other_compact));
        assert_eq!(compact.area(), 100.0);
    }
}

// ============================================================================
// 4. 使用示例
// ============================================================================

pub fn usage_example() {
    // 创建测试数据
    let pages = vec![
        (1, BoundingBox::new(0.0, 0.0, 10.0, 10.0)),
        (2, BoundingBox::new(10.0, 0.0, 20.0, 10.0)),
        (3, BoundingBox::new(0.0, 10.0, 10.0, 20.0)),
        (4, BoundingBox::new(10.0, 10.0, 20.0, 20.0)),
    ];
    
    // 构建空间树
    let tree = SpatialTree::build_from_pages(pages);
    
    // 执行查询
    let query = SpatialQuery::Intersects(BoundingBox::new(5.0, 5.0, 15.0, 15.0));
    let results = tree.search_pages(&query);
    
    println!("Found {} matching pages: {:?}", results.len(), results);
    
    // 获取统计信息
    let stats = tree.statistics();
    println!("Tree statistics: {:#?}", stats);
}

// ============================================================================
// 5. 迁移指南
// ============================================================================

/*
迁移步骤：

1. 替换 trait object 使用：
   - 将 Box<dyn GiSTPredicate> 替换为 BoundingBox
   - 将 Box<dyn GiSTQuery> 替换为 SpatialQuery

2. 更新搜索调用：
   - 旧：tree.search_pages(query, &ops)
   - 新：tree.search_pages(&query)

3. 简化操作：
   - 移除未使用的 GiSTOperations 方法
   - 直接使用 BoundingBox 的方法

4. 性能优化：
   - 考虑使用 CompactBoundingBox 减少内存占用
   - 使用新的并行搜索 API

5. 测试验证：
   - 确保所有现有测试通过
   - 添加新的性能基准测试
*/