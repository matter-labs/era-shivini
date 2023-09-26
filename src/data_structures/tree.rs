use super::*;

pub struct TreeCache {
    fri_lde_degree: usize,
    trace_subtrees: Vec<SubTree>,
    argument_subtrees: Vec<SubTree>,
    setup_subtrees: Vec<SubTree>,
    quotient_subtrees: Vec<SubTree>,
}

impl TreeCache {
    pub fn empty(fri_lde_degree: usize) -> Self {
        assert!(fri_lde_degree.is_power_of_two());
        Self {
            fri_lde_degree,
            trace_subtrees: vec![],
            argument_subtrees: vec![],
            setup_subtrees: vec![],
            quotient_subtrees: vec![],
        }
    }

    pub fn set_trace_subtrees(&mut self, trace_subtrees: Vec<SubTree>) {
        assert_eq!(trace_subtrees.len(), self.fri_lde_degree);

        self.trace_subtrees = trace_subtrees
    }

    pub fn get_trace_subtrees(&self) -> &Vec<SubTree> {
        assert_eq!(self.trace_subtrees.len(), self.fri_lde_degree);
        &self.trace_subtrees
    }

    pub fn set_argument_subtrees(&mut self, argument_subtrees: Vec<SubTree>) {
        assert_eq!(argument_subtrees.len(), self.fri_lde_degree);

        self.argument_subtrees = argument_subtrees
    }

    pub fn get_argument_subtrees(&self) -> &Vec<SubTree> {
        assert_eq!(self.argument_subtrees.len(), self.fri_lde_degree);
        &self.argument_subtrees
    }

    pub fn set_setup_subtrees(&mut self, setup_subtrees: Vec<SubTree>) {
        assert_eq!(setup_subtrees.len(), self.fri_lde_degree);

        self.setup_subtrees = setup_subtrees
    }

    pub fn get_setup_subtrees(&self) -> &Vec<SubTree> {
        assert_eq!(self.setup_subtrees.len(), self.fri_lde_degree);
        &self.setup_subtrees
    }

    pub fn set_quotient_subtrees(&mut self, quotient_subtrees: Vec<SubTree>) {
        assert_eq!(quotient_subtrees.len(), self.fri_lde_degree);

        self.quotient_subtrees = quotient_subtrees
    }

    pub fn get_quotient_subtrees(&self) -> &Vec<SubTree> {
        assert_eq!(self.quotient_subtrees.len(), self.fri_lde_degree);
        &self.quotient_subtrees
    }
}
