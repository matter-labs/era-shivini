use std::collections::HashMap;

use boojum::cs::oracle::merkle_tree::MerkleTreeWithCap;

use super::*;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum OracleType {
    Trace,
    Argument,
    Quotient,
    Setup,
}

impl OracleType {
    pub fn to_str(&self) -> &str {
        match self {
            OracleType::Trace => "Trace",
            OracleType::Argument => "Argument",
            OracleType::Quotient => "Quotient",
            OracleType::Setup => "Setup",
        }
    }
}

pub struct TreeCache {
    folding_schedule: Vec<usize>,
    fri_lde_degree: usize,
    trace_subtrees: Vec<SubTree>,
    argument_subtrees: Vec<SubTree>,
    setup_subtrees: Vec<SubTree>,
    quotient_subtrees: Vec<SubTree>,
    storage: HashMap<OracleType, Vec<SubTree>>,
}

impl TreeCache {
    pub fn empty(fri_lde_degree: usize) -> Self {
        assert!(fri_lde_degree.is_power_of_two());
        Self {
            folding_schedule: vec![],
            fri_lde_degree,
            trace_subtrees: vec![],
            argument_subtrees: vec![],
            setup_subtrees: vec![],
            quotient_subtrees: vec![],
            storage: HashMap::new(),
        }
    }

    pub fn set_trace_subtrees(&mut self, subtrees: Vec<SubTree>) {
        assert_eq!(subtrees.len(), self.fri_lde_degree);
        assert!(self.storage.insert(OracleType::Trace, subtrees).is_none())
    }

    pub fn get_trace_subtrees(&self) -> &Vec<SubTree> {
        self.get(OracleType::Trace)
    }

    pub fn get_trace_subtree(&self, coset_idx: usize) -> &SubTree {
        self.get_coset_subtree(OracleType::Trace, coset_idx)
    }

    pub fn set_argument_subtrees(&mut self, subtrees: Vec<SubTree>) {
        assert_eq!(subtrees.len(), self.fri_lde_degree);
        assert!(self
            .storage
            .insert(OracleType::Argument, subtrees)
            .is_none())
    }

    pub fn get_argument_subtrees(&self) -> &Vec<SubTree> {
        self.get(OracleType::Argument)
    }

    pub fn get_argument_subtree(&self, coset_idx: usize) -> &SubTree {
        self.get_coset_subtree(OracleType::Argument, coset_idx)
    }

    pub fn set_setup_tree_from_host_data(
        &mut self,
        setup_tree: &MerkleTreeWithCap<F, DefaultTreeHasher>,
    ) {
        unimplemented!()
    }

    pub fn setup_setup_subtrees(&mut self, subtrees: Vec<SubTree>) {
        assert_eq!(subtrees.len(), self.fri_lde_degree);
        assert!(self.storage.insert(OracleType::Setup, subtrees).is_none())
    }

    pub fn get_setup_subtrees(&self) -> &Vec<SubTree> {
        self.get(OracleType::Setup)
    }

    pub fn get_setup_subtree(&self, coset_idx: usize) -> &SubTree {
        self.get_coset_subtree(OracleType::Setup, coset_idx)
    }

    pub fn set_quotient_subtrees(&mut self, subtrees: Vec<SubTree>) {
        assert_eq!(subtrees.len(), self.fri_lde_degree);
        assert!(self
            .storage
            .insert(OracleType::Quotient, subtrees)
            .is_none())
    }

    pub fn get_quotient_subtrees(&self) -> &Vec<SubTree> {
        self.get(OracleType::Quotient)
    }

    pub fn get_quotient_subtree(&self, coset_idx: usize) -> &SubTree {
        self.get_coset_subtree(OracleType::Quotient, coset_idx)
    }

    fn get_coset_subtree(&self, key: OracleType, coset_idx: usize) -> &SubTree {
        &self
            .storage
            .get(&key)
            .and_then(|oracle| oracle.get(coset_idx))
            .expect(key.to_str())
    }

    fn get(&self, key: OracleType) -> &Vec<SubTree> {
        &self.storage.get(&key).expect(key.to_str())
    }
}
