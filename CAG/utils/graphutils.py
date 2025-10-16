from pycparser import c_parser, c_ast, parse_file
from typing import List, Set, Tuple, Dict, Optional, Any
from datautils.classes import Node, Edge, AbstractGraph, CompactAbstractGraph
from collections import defaultdict
import tempfile
import os

class CASTVisitor(c_ast.NodeVisitor):
    """
    Custom visitor for pycparser AST to extract detailed information
    """
    def __init__(self):
        self.tokens = []
        self.properties = []
        self.ast_structure = []
        
    def visit(self, node):
        """Override visit to track structure"""
        node_info = {
            'type': node.__class__.__name__,
            'coord': node.coord if hasattr(node, 'coord') else None,
            'children': []
        }
        self.ast_structure.append(node_info)
        super().visit(node)
        
    def generic_visit(self, node):
        """Visit all node types"""
        # Extract tokens (leaf nodes with actual code values)
        if hasattr(node, 'name') and isinstance(node.name, str):
            self.tokens.append(('identifier', node.name))
        if hasattr(node, 'value') and isinstance(node.value, str):
            self.tokens.append(('literal', node.value))
        if hasattr(node, 'op') and isinstance(node.op, str):
            self.tokens.append(('operator', node.op))
        
        # Continue traversal
        for child in node:
            self.visit(child)
            
    # Additional methods to extract properties

class CCodeCAGBuilder:
    def __init__(self, use_cpp: bool = True, cpp_path: str = 'gcc', cpp_args: List[str] = None):
        """
        Initialize CAG builder
        
        Args:
            use_cpp: Whether to use C preprocessor
            cpp_path: Path to C preprocessor (gcc or clang)
            cpp_args: Additional preprocessor arguments
        """
        self.parser = c_parser.CParser()
        self.use_cpp = use_cpp
        self.cpp_path = cpp_path
        self.cpp_args = cpp_args or ['-E', '-I/usr/include', '-I/usr/local/include']
        self.node_counter = 0
        self.node_map: Dict[int, Node] = {}
        
        # C-specific mergeable aggregation types (based on C syntax)
        self.mergeable_aggregations = {
            # Expression aggregations
            'FuncCall', 'ArrayRef', 'BinaryOp', 'UnaryOp', 'TernaryOp',
            'Cast', 'Assignment', 'StructRef', 'ExprList',
            
            # Statement aggregations (only return is mergeable per paper)
            'Return',
            
            # Declaration aggregations
            'Decl', 'InitList',
            
            # Type aggregations
            'ArrayDecl', 'PtrDecl',
            
            # Other mergeable structures
            'ParamList', 'IdentifierType'
        }
        
        # Non-mergeable aggregations (preserve parallel structure)
        self.non_mergeable_aggregations = {
            'Compound',  # Statement block
            'If', 'For', 'While', 'DoWhile', 'Switch',
            'FuncDef'  # Function definition
        }
    
    def create_node(self, label: str, node_type: str, original_labels: List[str] = None) -> Node:
        """Create a new node with unique ID"""
        node = Node(
            id=self.node_counter,
            label=label,
            node_type=node_type,
            original_labels=original_labels or [label]
        )
        self.node_map[self.node_counter] = node
        self.node_counter += 1
        return node
    
    def parse_c_code(self, code: str, filename: str = '<string>') -> c_ast.FileAST:
        """
        Parse C code into AST using pycparser
        
        Args:
            code: C source code string
            filename: Optional filename for error messages
            
        Returns:
            pycparser AST
        """
        if self.use_cpp:
            # Write code to temporary file for preprocessing
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Parse with preprocessing
                ast = parse_file(
                    temp_file,
                    use_cpp=True,
                    cpp_path=self.cpp_path,
                    cpp_args=self.cpp_args
                )
            finally:
                os.unlink(temp_file)
        else:
            # Direct parsing without preprocessing
            ast = self.parser.parse(code, filename=filename)
        
        return ast
    
    def extract_function_asts(self, file_ast: c_ast.FileAST) -> List[Tuple[str, c_ast.FuncDef]]:
        """
        Extract individual function ASTs from file AST
        
        Returns:
            List of (function_name, function_ast) tuples
        """
        functions = []
        
        for node in file_ast.ext:
            if isinstance(node, c_ast.FuncDef):
                func_name = node.decl.name
                functions.append((func_name, node))
        
        return functions
    
    def ast_to_graph_nodes(self, ast_node: c_ast.Node, parent_graph_node: Optional[Node] = None,
                          ag: AbstractGraph = None) -> Node:
        """
        Recursively convert pycparser AST to Abstract Graph nodes
        This creates the reversed edge structure as per paper Section 3.1
        
        Args:
            ast_node: Current AST node from pycparser
            parent_graph_node: Parent node in the graph (reversed from AST)
            ag: Abstract Graph being built
            
        Returns:
            Created graph node
        """
        node_type_name = ast_node.__class__.__name__
        
        # Determine if this is a token node or property node
        is_token = False
        token_value = None
        
        # Extract token values from leaf nodes
        if isinstance(ast_node, c_ast.ID):
            is_token = True
            token_value = ast_node.name
        elif isinstance(ast_node, c_ast.Constant):
            is_token = True
            token_value = ast_node.value
        elif hasattr(ast_node, 'op') and isinstance(ast_node.op, str):
            # Operators
            is_token = True
            token_value = ast_node.op
        elif isinstance(ast_node, c_ast.IdentifierType):
            # Type names like 'int', 'char', etc.
            if ast_node.names:
                is_token = True
                token_value = ' '.join(ast_node.names)
        
        # Create graph node
        if is_token:
            current_node = self.create_node(token_value, 'token')
            ag.token_nodes.append(current_node)
        else:
            current_node = self.create_node(node_type_name, 'property')
            ag.property_nodes.add(current_node)
        
        ag.nodes.add(current_node)
        
        # Create reversed edge (child -> parent, opposite of AST direction)
        if parent_graph_node:
            ag.edges.add(Edge(current_node, parent_graph_node, 'ast'))
        
        # Recursively process children
        for child_name, child in ast_node.children():
            if child is not None:
                # Create property node for the relationship if needed
                if not is_token:  # Only add relationship nodes for non-tokens
                    rel_node = self.create_node(child_name, 'property')
                    ag.nodes.add(rel_node)
                    ag.property_nodes.add(rel_node)
                    ag.edges.add(Edge(rel_node, current_node, 'ast'))
                    
                    # Process child with relationship node as parent
                    self.ast_to_graph_nodes(child, rel_node, ag)
                else:
                    # For token nodes, skip intermediate relationship
                    self.ast_to_graph_nodes(child, current_node, ag)
        
        return current_node
    
    def build_abstract_graph(self, func_ast: c_ast.FuncDef) -> AbstractGraph:
        """
        Stage 1: Convert C function AST to Abstract Graph (AG)
        Implements Section 3.1 of the paper
        
        Args:
            func_ast: pycparser FuncDef AST node
            
        Returns:
            Abstract Graph with reversed edges and additional connections
        """
        ag = AbstractGraph()
        
        # Create sink node (root of function)
        func_name = func_ast.decl.name if func_ast.decl else "unknown"
        sink = self.create_node(f'FuncDef_{func_name}', 'sink')
        ag.sink = sink
        ag.nodes.add(sink)
        
        # Convert AST to graph with reversed edges
        self.ast_to_graph_nodes(func_ast, sink, ag)
        
        # Add edges from each token to the next token (sequence order)
        for i in range(len(ag.token_nodes) - 1):
            ag.edges.add(Edge(ag.token_nodes[i], ag.token_nodes[i + 1], 'next_token'))
        
        # Add edges from each token directly to sink
        for token in ag.token_nodes:
            ag.edges.add(Edge(token, sink, 'to_sink'))
        
        return ag
    
    def find_single_entry_sequences(self, ag: AbstractGraph) -> List[List[Node]]:
        """
        Find all longest sequences of single-entry property nodes
        Implements Section 3.2 of the proposed method
        
        A sequence is a chain where each node has exactly one incoming edge
        """
        sequences = []
        visited = set()
        
        # Build adjacency information
        in_degree = defaultdict(int)
        out_edges = defaultdict(list)
        
        for edge in ag.edges:
            in_degree[edge.target] += 1
            out_edges[edge.source].append(edge)
        
        # Find sequences starting from each unvisited single-entry property node
        for node in ag.property_nodes:
            if node in visited or node == ag.sink:
                continue
            
            if in_degree[node] != 1:
                continue
            
            # Check if this could be start of a sequence
            # Start must connect from a token node or multi-entry property node
            incoming = ag.get_incoming_edges(node)
            if not incoming:
                continue
            
            source = incoming[0].source
            if source.node_type != 'token' and in_degree[source] == 1:
                continue  # Not a valid sequence start
            
            # Build sequence
            sequence = []
            current = node
            
            while current and current in ag.property_nodes and current != ag.sink:
                if current in visited:
                    break
                
                if in_degree[current] != 1:
                    break
                
                sequence.append(current)
                visited.add(current)
                
                # Find next node in sequence
                outgoing = out_edges[current]
                next_node = None
                
                for edge in outgoing:
                    target = edge.target
                    if target in ag.property_nodes and target != ag.sink:
                        if in_degree[target] == 1:
                            next_node = target
                            break
                
                current = next_node
            
            # Only keep sequences with 2+ nodes
            if len(sequence) >= 2:
                sequences.append(sequence)
        
        return sequences
    
    def merge_node_sequence(self, ag: AbstractGraph, sequence: List[Node]) -> Node:
        """
        Merge a sequence of single-entry nodes into one node
        Preserves incoming and outgoing connections
        """
        # Create merged node with all labels
        merged_label = '->'.join([n.label for n in sequence])
        original_labels = []
        for node in sequence:
            original_labels.extend(node.original_labels)
        
        merged = self.create_node(merged_label, 'merged_sequence', original_labels)
        
        # Find connections
        first_node = sequence[0]
        last_node = sequence[-1]
        
        incoming = ag.get_incoming_edges(first_node)
        outgoing = ag.get_outgoing_edges(last_node)
        
        # Remove old nodes and their edges
        sequence_set = set(sequence)
        ag.nodes -= sequence_set
        ag.property_nodes -= sequence_set
        
        edges_to_remove = set()
        for edge in ag.edges:
            if edge.source in sequence_set or edge.target in sequence_set:
                edges_to_remove.add(edge)
        ag.edges -= edges_to_remove
        
        # Add merged node
        ag.nodes.add(merged)
        ag.property_nodes.add(merged)
        
        # Reconnect
        for edge in incoming:
            if edge.source not in sequence_set:
                ag.edges.add(Edge(edge.source, merged, edge.edge_type))
        
        for edge in outgoing:
            if edge.target not in sequence_set:
                ag.edges.add(Edge(merged, edge.target, edge.edge_type))
        
        return merged
    
    def find_aggregation_structures(self, ag: AbstractGraph) -> List[Tuple[Node, List[Node]]]:
        """
        Find mergeable aggregation structures
        Implements Section 3.3 of the paper
        
        An aggregation is a parent node with 2+ children
        Only specific types are mergeable, these specific types are defined in the paper Yu Luo et al. Table 1.
        """
        in_degree = defaultdict(int)
        for edge in ag.edges:
            in_degree[edge.target] += 1
        
        aggregations = []
        
        for parent in ag.property_nodes:
            if parent == ag.sink:
                continue
            
            # Must have 2+ incoming edges
            if in_degree[parent] < 2:
                continue
            
            # Check if this type is mergeable
            is_mergeable = False
            for mergeable_type in self.mergeable_aggregations:
                if mergeable_type in parent.label:
                    is_mergeable = True
                    break
            
            if not is_mergeable:
                continue
            
            # Find all children (nodes pointing to parent)
            children = [e.source for e in ag.edges if e.target == parent]
            
            if len(children) < 2:
                continue
            
            # Verify all children have single entry (from aggregation perspective)
            # Children should not be parent nodes themselves (would lose structure)
            valid_children = []
            for child in children:
                if child in ag.property_nodes:
                    # Property nodes are OK if they have single entry
                    if in_degree[child] == 1:
                        valid_children.append(child)
                else:
                    # Token nodes are always OK
                    valid_children.append(child)
            
            if len(valid_children) == len(children) and len(valid_children) >= 2:
                aggregations.append((parent, valid_children))
        
        return aggregations
    
    def merge_aggregation(self, ag: AbstractGraph, parent: Node, children: List[Node]) -> Node:
        """
        Merge an aggregation structure (parent + children) into single node
        Implements Algorithm 1 from the paper
        """
        # Create merged node
        all_labels = [parent.label] + [c.label for c in children]
        merged_label = '{' + ','.join(all_labels) + '}'
        
        original_labels = parent.original_labels.copy()
        for child in children:
            original_labels.extend(child.original_labels)
        
        merged = self.create_node(merged_label, 'merged_aggregation', original_labels)
        
        # Find connections
        # Alpha: nodes connecting TO children (excluding internal edges)
        alpha_edges = []
        children_set = set(children)
        for child in children:
            for edge in ag.get_incoming_edges(child):
                if edge.source not in children_set and edge.source != parent:
                    alpha_edges.append(edge)
        
        # Omega: nodes connected FROM parent
        omega_edges = ag.get_outgoing_edges(parent)
        
        # Remove old nodes and edges
        nodes_to_remove = {parent} | children_set
        ag.nodes -= nodes_to_remove
        ag.property_nodes -= nodes_to_remove
        
        edges_to_remove = set()
        for edge in ag.edges:
            if edge.source in nodes_to_remove or edge.target in nodes_to_remove:
                edges_to_remove.add(edge)
        ag.edges -= edges_to_remove
        
        # Add merged node
        ag.nodes.add(merged)
        ag.property_nodes.add(merged)
        
        # Reconnect
        for edge in alpha_edges:
            ag.edges.add(Edge(edge.source, merged, edge.edge_type))
        
        for edge in omega_edges:
            ag.edges.add(Edge(merged, edge.target, edge.edge_type))
        
        return merged
    
    def build_cag_from_function(self, func_ast: c_ast.FuncDef) -> CompactAbstractGraph:
        """
        Complete pipeline: Function AST -> AG -> CAG
        
        Args:
            func_ast: pycparser FuncDef AST
            
        Returns:
            Compact Abstract Graph
        """
        # Build Abstract Graph
        ag = self.build_abstract_graph(func_ast)
        
        # Stage 1: Merge single-entry sequences
        sequences = self.find_single_entry_sequences(ag)
        for sequence in sequences:
            self.merge_node_sequence(ag, sequence)
        
        # Stage 2: Merge aggregation structures
        # Iterate until no more aggregations can be merged
        iteration = 0
        max_iterations = 10 
        
        while iteration < max_iterations:
            aggregations = self.find_aggregation_structures(ag)
            if not aggregations:
                break
            
            for parent, children in aggregations:
                # Verify nodes still exist (may have been merged)
                if parent in ag.nodes and all(c in ag.nodes for c in children):
                    self.merge_aggregation(ag, parent, children)
            
            iteration += 1
        
        # Convert to CAG
        cag = CompactAbstractGraph(
            nodes=ag.nodes.copy(),
            edges=ag.edges.copy(),
            sink=ag.sink
        )
        
        return cag
    
    def build_cag_from_code(self, code: str) -> Dict[str, CompactAbstractGraph]:
        """
        Build CAGs for all functions in C code
        
        Args:
            code: C source code string
            
        Returns:
            Dictionary mapping function names to their CAGs
        """
        # Parse code
        file_ast = self.parse_c_code(code)
        
        # Extract functions
        functions = self.extract_function_asts(file_ast)
        
        # Build CAG for each function
        cags = {}
        for func_name, func_ast in functions:
            # Reset node counter for each function (optional, for cleaner IDs)
            # self.node_counter = 0
            # self.node_map = {}
            
            cag = self.build_cag_from_function(func_ast)
            cags[func_name] = cag
        
        return cags
    
