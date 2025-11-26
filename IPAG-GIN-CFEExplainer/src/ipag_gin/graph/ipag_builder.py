from tree_sitter import Parser
import pandas as pd
import numpy as np
from collections import Counter

class IPAGBuilder:
    """
    Builds Abstract Syntax Trees (ASTs) and Interprocedural Analysis Graphs (IPAGs)
    for code snippets in various programming languages.
    """
    
    def __init__(self, source=None, language=None, lang_map=None):
        """
        Initialize the IPAG Builder.
        
        Args:
            source (pd.Series): Series containing code snippets
            language (pd.Series): Series containing language identifiers for each code snippet
            lang_map (dict): Dictionary mapping language names to tree-sitter language objects
        
        Raises:
            ValueError: If lang_map is None or if source and language have different lengths
        """
        if lang_map is None:
            raise ValueError("Please provide a valid language map")
        
        # Initialize with empty Series if not provided
        if source is None:
            source = pd.Series(dtype=str)
        if language is None:
            language = pd.Series(dtype=str)
            
        # Validate that source and language have the same length
        if len(source) != len(language):
            raise ValueError(
                f"Source and language must have the same length. "
                f"Got source: {len(source)}, language: {len(language)}"
            )
        
        # Ensure inputs are Series
        if not isinstance(source, pd.Series):
            raise TypeError("Source must be a pandas Series")
        if not isinstance(language, pd.Series):
            raise TypeError("Language must be a pandas Series")
        
        self._code = source.reset_index(drop=True)
        self._lang = language.reset_index(drop=True)
        self._lang_map = lang_map
        self._ast_parser = Parser()
        self._asts = []
        self.ipags = pd.DataFrame()

    def _build_ast(self, code, lang):
        """
        Build AST for a single code snippet.
        
        Args:
            code (str): Source code to parse
            lang (str): Programming language identifier
            
        Returns:
            tree_sitter.Tree: Parsed AST tree
            
        Raises:
            ValueError: If language is not in language map
        """
        if lang not in self._lang_map:
            raise ValueError(
                f"Language '{lang}' not found in language map. "
                f"Available languages: {list(self._lang_map.keys())}"
            )
        
        lang_obj = self._lang_map[lang]
        self._ast_parser.language=lang_obj
        
        # Ensure code is string and encode to bytes
        code_str = str(code) if code is not None else ""
        tree = self._ast_parser.parse(bytes(code_str, "utf8"))
        self._ast_parser.reset()
        return tree

    def build_all_asts(self):
        """
        Build ASTs for all code snippets in the source Series.
        Stores results in self._asts list.
        """
        print("=" * 60)
        print("Building ASTs for all code snippets")
        print("=" * 60)
        
        self._asts = []
        successful = 0
        failed = 0
        
        for idx in range(len(self._code)):
            code = str(self._code.iloc[idx])
            lang = str(self._lang.iloc[idx])
            
            try:
                ast = self._build_ast(code, lang)
                self._asts.append(ast)
                successful += 1
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(self._code)} snippets...")
                    
            except Exception as e:
                print(f"Error processing snippet at index {idx} (language: {lang}): {str(e)}")
                self._asts.append(None)
                failed += 1
        
        print("=" * 60)
        print(f"Finished building ASTs")
        print(f"Successful: {successful}, Failed: {failed}, Total: {len(self._code)}")
        print("=" * 60)
    
    def _map_ast_node_to_ipag(self, ast_node, lang):
        """
        Map a single AST node to IPAG node type: 'TOKEN', 'DECLARATION', or 'PROPERTY'
        
        Args:
            ast_node: dict with fields 'type', 'label', 'id' (AST parser output)
            lang: 'py', 'java', 'c', 'cpp'
        
        Returns:
            tuple: (node_type (str), label (str))
        """
        # Map short language codes to internal names
        lang_map = {
            'py': 'python',
            'c': 'c',
            'cpp': 'cpp',
            'java': 'java',
        }
        
        if lang not in lang_map:
            raise ValueError(f"Unsupported language: {lang}. Supported: py, c, cpp, java")
        
        normalized_lang = lang_map[lang]
        
        type_map = {
            'python': {
                'token_types': {
                    # Identifiers and literals
                    'Name', 'Identifier', 'Constant', 'Str', 'Num', 'Bytes',
                    'NameConstant', 'Ellipsis', 'True', 'False', 'None',
                    # Operators
                    'Add', 'Sub', 'Mult', 'Div', 'Mod', 'Pow', 'LShift', 'RShift',
                    'BitOr', 'BitXor', 'BitAnd', 'FloorDiv', 'MatMult',
                    'And', 'Or', 'Not', 'Invert', 'UAdd', 'USub',
                    'Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE', 'Is', 'IsNot', 'In', 'NotIn',
                    # Keywords
                    'keyword', 'alias', 'arg',
                },
                'decl_types': {
                    # Function and class definitions
                    'FunctionDef', 'AsyncFunctionDef', 'ClassDef', 'Lambda',
                    # Variable assignments
                    'Assign', 'AnnAssign', 'AugAssign', 'NamedExpr',
                    # Import statements
                    'Import', 'ImportFrom',
                    # Type definitions
                    'TypeAlias',
                },
            },
            'java': {
                'token_types': {
                    # Identifiers and literals
                    'Identifier', 'SimpleName', 'QualifiedName',
                    'StringLiteral', 'NumberLiteral', 'CharacterLiteral', 
                    'BooleanLiteral', 'NullLiteral', 'TextBlock',
                    # Literals
                    'Literal', 'IntegerLiteral', 'FloatingPointLiteral',
                    # Operators
                    'Operator', 'InfixExpression', 'PrefixExpression', 'PostfixExpression',
                    # Keywords
                    'Keyword', 'Modifier', 'PrimitiveType',
                },
                'decl_types': {
                    # Type declarations
                    'TypeDeclaration', 'ClassDeclaration', 'InterfaceDeclaration',
                    'EnumDeclaration', 'AnnotationTypeDeclaration', 'RecordDeclaration',
                    # Member declarations
                    'MethodDeclaration', 'ConstructorDeclaration', 'FieldDeclaration',
                    'EnumConstantDeclaration', 'AnnotationTypeMemberDeclaration',
                    # Variable declarations
                    'VariableDeclarationStatement', 'VariableDeclarationExpression',
                    'VariableDeclarationFragment', 'SingleVariableDeclaration',
                    # Package and imports
                    'PackageDeclaration', 'ImportDeclaration',
                },
            },
            'c': {
                'token_types': {
                    # Identifiers and constants
                    'Identifier', 'identifier', 'DeclRefExpr',
                    'IntegerLiteral', 'FloatingLiteral', 'CharacterLiteral',
                    'StringLiteral', 'Constant',
                    # Operators
                    'Operator', 'BinaryOperator', 'UnaryOperator',
                    'UnaryExprOrTypeTraitExpr',
                    # Type references
                    'BuiltinType', 'TypeRef',
                },
                'decl_types': {
                    # Function declarations
                    'FunctionDecl', 'function_definition', 'function_declarator',
                    # Variable declarations
                    'VarDecl', 'ParmVarDecl', 'declaration', 'init_declarator',
                    # Type declarations
                    'TypedefDecl', 'typedef_declaration', 'type_definition',
                    'RecordDecl', 'struct_specifier', 'union_specifier',
                    'EnumDecl', 'enum_specifier', 'enumerator',
                    # Preprocessor
                    'preproc_def', 'preproc_function_def', 'preproc_include',
                },
            },
            'cpp': {
                'token_types': {
                    # Identifiers and constants
                    'Identifier', 'identifier', 'DeclRefExpr',
                    'IntegerLiteral', 'FloatingLiteral', 'CharacterLiteral',
                    'StringLiteral', 'CXXBoolLiteralExpr', 'CXXNullPtrLiteralExpr',
                    'Constant', 'UserDefinedLiteral',
                    # Operators
                    'Operator', 'BinaryOperator', 'UnaryOperator',
                    'CXXOperatorCallExpr', 'UnaryExprOrTypeTraitExpr',
                    # Type references
                    'BuiltinType', 'TypeRef', 'TemplateRef',
                },
                'decl_types': {
                    # Function declarations
                    'FunctionDecl', 'CXXMethod', 'CXXConstructor', 'CXXDestructor',
                    'FunctionTemplate', 'function_definition', 'function_declarator',
                    # Variable declarations
                    'VarDecl', 'ParmVarDecl', 'FieldDecl', 'field_declaration',
                    'declaration', 'init_declarator',
                    # Type declarations
                    'TypedefDecl', 'TypeAliasDecl', 'typedef_declaration',
                    'ClassDecl', 'CXXRecord', 'class_specifier', 'struct_specifier',
                    'EnumDecl', 'enum_specifier', 'enumerator',
                    'UnionDecl', 'union_specifier',
                    # Templates
                    'ClassTemplate', 'template_declaration',
                    # Namespace
                    'NamespaceDecl', 'namespace_definition', 'UsingDirective',
                    'UsingDeclaration', 'NamespaceAlias', 'using_declaration',
                    # Preprocessor
                    'preproc_def', 'preproc_function_def', 'preproc_include',
                },
            }
        }
        
        # Additional property types that represent structural elements
        property_types = {
            'python': {
                'Module', 'Expression', 'Interactive', 'FunctionType',
                'If', 'While', 'For', 'AsyncFor', 'With', 'AsyncWith',
                'Try', 'ExceptHandler', 'Match', 'match_case',
                'Return', 'Delete', 'Yield', 'YieldFrom', 'Raise',
                'Assert', 'Pass', 'Break', 'Continue', 'Global', 'Nonlocal',
                'Expr', 'Call', 'Attribute', 'Subscript', 'Starred',
                'List', 'Tuple', 'Set', 'Dict', 'ListComp', 'SetComp', 
                'DictComp', 'GeneratorExp', 'Compare', 'BoolOp', 'BinOp',
                'UnaryOp', 'IfExp', 'JoinedStr', 'FormattedValue',
                'comprehension', 'arguments', 'withitem',
            },
            'java': {
                'CompilationUnit', 'Block', 'Statement',
                'ExpressionStatement', 'ReturnStatement', 'IfStatement',
                'WhileStatement', 'DoStatement', 'ForStatement', 'EnhancedForStatement',
                'SwitchStatement', 'SwitchCase', 'TryStatement', 'CatchClause',
                'ThrowStatement', 'SynchronizedStatement', 'AssertStatement',
                'BreakStatement', 'ContinueStatement', 'LabeledStatement',
                'MethodInvocation', 'SuperMethodInvocation', 'ClassInstanceCreation',
                'ArrayCreation', 'ArrayAccess', 'FieldAccess', 'SuperFieldAccess',
                'Assignment', 'ConditionalExpression', 'InstanceofExpression',
                'CastExpression', 'LambdaExpression', 'MethodReference',
                'ArrayType', 'ParameterizedType', 'WildcardType', 'UnionType',
                'Dimension', 'TypeParameter', 'ArrayInitializer',
            },
            'c': {
                'TranslationUnitDecl', 'compound_statement', 'expression_statement',
                'if_statement', 'switch_statement', 'case_statement', 'labeled_statement',
                'while_statement', 'do_statement', 'for_statement',
                'return_statement', 'break_statement', 'continue_statement', 'goto_statement',
                'CallExpr', 'call_expression', 'MemberExpr', 'field_expression',
                'ArraySubscriptExpr', 'subscript_expression',
                'ParenExpr', 'parenthesized_expression',
                'CStyleCastExpr', 'cast_expression',
                'ConditionalOperator', 'conditional_expression',
                'CompoundStmt', 'InitListExpr', 'initializer_list',
                'pointer_declarator', 'array_declarator', 'parameter_list',
                'argument_list', 'field_declaration_list',
            },
            'cpp': {
                'TranslationUnitDecl', 'compound_statement', 'expression_statement',
                'if_statement', 'switch_statement', 'case_statement', 'labeled_statement',
                'while_statement', 'do_statement', 'for_statement', 'range_based_for_statement',
                'return_statement', 'break_statement', 'continue_statement', 'goto_statement',
                'try_statement', 'catch_clause', 'throw_statement',
                'CallExpr', 'CXXMemberCallExpr', 'call_expression',
                'MemberExpr', 'field_expression', 'CXXThisExpr',
                'ArraySubscriptExpr', 'subscript_expression',
                'ParenExpr', 'parenthesized_expression',
                'CXXStaticCastExpr', 'CXXDynamicCastExpr', 'CXXReinterpretCastExpr',
                'CXXConstCastExpr', 'CStyleCastExpr', 'cast_expression',
                'ConditionalOperator', 'conditional_expression',
                'CXXNewExpr', 'new_expression', 'CXXDeleteExpr', 'delete_expression',
                'CXXThrowExpr', 'LambdaExpr', 'lambda_expression',
                'CompoundStmt', 'InitListExpr', 'initializer_list',
                'CXXConstructExpr', 'template_argument_list',
                'pointer_declarator', 'reference_declarator', 'array_declarator',
                'parameter_list', 'argument_list', 'field_declaration_list',
                'base_class_clause', 'access_specifier',
            }
        }

        
        # Extract node type and label
        tp = ast_node.get('type', '')
        label = ast_node.get('label', '')
        
        # Determine IPAG node type
        if tp in type_map[normalized_lang]['token_types']:
            return 'TOKEN', label
        elif tp in type_map[normalized_lang]['decl_types']:
            return 'DECLARATION', label
        elif tp in property_types.get(normalized_lang, set()):
            return 'PROPERTY', label
        else:
            # Default to PROPERTY for unknown types
            return 'PROPERTY', label

    def _extract_nodes_and_edges_from_tree_sitter(self):
        """
        Extract all nodes and edges from tree-sitter tree objects.
        
        Returns:
            tuple: (ast_nodes, ast_edges)
                - ast_nodes: List of List of node dicts with 'id', 'type', 'label' fields
                - ast_edges: List of List of edge dicts with 'source', 'target', 'type' fields
        """
        all_ast_nodes = []
        all_ast_edges = []
        
        for idx in range(len(self._code)):
            source_code = self._code.iloc[idx]
            tree = self._asts[idx]
            
            if tree is None:
                all_ast_nodes.append([])
                all_ast_edges.append([])
                continue
                
            if isinstance(source_code, str):
                source_code = source_code.encode('utf8')
            
            nodes = []
            edges = []
            node_id = 0
            
            def traverse(node, parent_id=None, depth=0):
                nonlocal node_id
                
                current_id = node_id
                
                # Extract text content for the node
                text = source_code[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
                
                # For leaf nodes (terminals), use the text as label
                # For non-leaf nodes, use empty label or node type
                if node.child_count == 0:
                    label = text.strip()
                else:
                    # For named nodes like function definitions, try to extract the name
                    label = ''
                    # Look for identifier child nodes
                    for child in node.children:
                        if child.type == 'identifier' or child.type == 'Identifier':
                            label = source_code[child.start_byte:child.end_byte].decode('utf8', errors='ignore')
                            break
                
                # Create node dict
                node_dict = {
                    'id': str(current_id),
                    'type': node.type,
                    'label': label,
                    'start_byte': node.start_byte,
                    'end_byte': node.end_byte,
                    'start_point': node.start_point,
                    'end_point': node.end_point,
                }
                nodes.append(node_dict)
                node_id += 1
                
                # Create edge from parent to current node (AST structural edge)
                if parent_id is not None:
                    edges.append({
                        'source': str(parent_id),
                        'target': str(current_id),
                        'type': 'AST_EDGE'
                    })
                
                # Recursively traverse children
                for child in node.children:
                    traverse(child, current_id, depth + 1)
            
            # Start traversal from root
            traverse(tree.root_node)
            all_ast_nodes.append(nodes)
            all_ast_edges.append(edges)
        
        return all_ast_nodes, all_ast_edges

    def _process_ast_to_ipag(self):
        """
        Convert AST trees to IPAG nodes and edges.
        
        Returns:
            tuple: (ipag_nodes_repo, ipag_edges_repo)
                - ipag_nodes_repo: List of List of IPAG node dicts
                - ipag_edges_repo: List of List of IPAG edge dicts
        """
        ast_nodes_repo, ast_edges_repo = self._extract_nodes_and_edges_from_tree_sitter()
        ipag_nodes_repo = []
        ipag_edges_repo = []
        
        for idx in range(len(ast_nodes_repo)):
            ast_nodes = ast_nodes_repo[idx]
            ast_edges = ast_edges_repo[idx]
            lang = self._lang.iloc[idx]
            
            ipag_nodes = []
            # Process nodes
            for node in ast_nodes:
                try:
                    node_type, label = self._map_ast_node_to_ipag(node, lang)
                    ipag_nodes.append({
                        'id': node.get('id', ''),
                        'type': node_type,
                        'label': label,
                        'original_type': node.get('type', '')  # Preserve original AST type
                    })
                except Exception as e:
                    # Handle errors gracefully, default to PROPERTY
                    print(f"Warning: Error processing node {node.get('id', 'unknown')}: {e}")
                    ipag_nodes.append({
                        'id': node.get('id', ''),
                        'type': 'PROPERTY',
                        'label': node.get('label', ''),
                        'original_type': node.get('type', '')
                    })
            
            # Process edges - keep the same structure but now in IPAG
            ipag_edges = []
            for edge in ast_edges:
                ipag_edges.append({
                    'source': edge['source'],
                    'target': edge['target'],
                    'type': 'CHILD'  # AST parent-child relationship
                })
            

            ## This step is only for multi class classification
            # 2. Add CONTROL_FLOW edges
            cf_edges = self._add_control_flow_edges(ipag_nodes, ast_edges, lang)
            ipag_edges.extend(cf_edges)
            
            # 3. Add NEXT_TOKEN edges
            next_edges = self._add_next_token_edges(ipag_nodes)
            ipag_edges.extend(next_edges)
            
            ipag_nodes_repo.append(ipag_nodes)
            ipag_edges_repo.append(ipag_edges)


        return ipag_nodes_repo, ipag_edges_repo

    
    def build(self):
        """Build ASTs and convert to IPAGs with nodes and edges."""
        self.build_all_asts()
        nodes, edges = self._process_ast_to_ipag()
        
        # Store results
        self.ipag_nodes = nodes
        self.ipag_edges = edges
        
        print("=" * 60)
        print("IPAG Construction Complete")
        print(f"Total snippets: {len(nodes)}")
        print(f"Average nodes per snippet: {sum(len(n) for n in nodes) / len(nodes) if nodes else 0:.1f}")
        print(f"Average edges per snippet: {sum(len(e) for e in edges) / len(edges) if edges else 0:.1f}")
        print("=" * 60)
        
        return nodes, edges

    def get_ipag_dataframe(self):
        """
        Create a DataFrame with code, language, IPAG nodes, and edges.
        
        Returns:
            pd.DataFrame: DataFrame containing code snippets with their IPAG representations
        """
        if not hasattr(self, 'ipag_nodes') or not hasattr(self, 'ipag_edges'):
            print("Warning: IPAGs not built yet. Call build() first.")
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'code': self._code,
            'language': self._lang,
            'ipag_nodes': self.ipag_nodes,
            'ipag_edges': self.ipag_edges,
            'num_nodes': [len(nodes) for nodes in self.ipag_nodes],
            'num_edges': [len(edges) for edges in self.ipag_edges]
        })
        
        return df

    def get_asts(self):
        """
        Get the list of built ASTs.
        
        Returns:
            list: List of tree_sitter.Tree objects (or None for failed parses)
        """
        if not self._asts:
            print("Warning: No ASTs have been built yet. Call build_all_asts() first.")
        return self._asts

    def get_ast_summary(self):
        """
        Get a summary of the AST building results.
        
        Returns:
            dict: Dictionary containing statistics about the ASTs
        """
        total = len(self._asts)
        successful = sum(1 for ast in self._asts if ast is not None)
        failed = total - successful
        
        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0
        }

    def get_ast_dataframe(self):
        """
        Create a DataFrame with code, language, and AST information.
        
        Returns:
            pd.DataFrame: DataFrame containing code snippets and their AST status
        """
        if not self._asts:
            print("Warning: No ASTs have been built yet. Call build_all_asts() first.")
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'code': self._code,
            'language': self._lang,
            'ast': self._asts,
            'has_ast': [ast is not None for ast in self._asts]
        })
        
        return df
    

    def _add_control_flow_edges(self, nodes, edges, lang):
        """
        Add control flow edges (e.g., if -> then, loop -> body).
        Based on node types, not full CFG construction.
        """
        cfg_edges = []
        
        # Map node types to control flow patterns
        cf_patterns = {
            'python': {
                'If': ['test', 'body', 'orelse'],
                'While': ['test', 'body', 'orelse'],
                'For': ['target', 'iter', 'body', 'orelse'],
                'Try': ['body', 'handlers', 'orelse', 'finalbody'],
            },
            'c': {
                'if_statement': ['condition', 'consequence', 'alternative'],
                'while_statement': ['condition', 'body'],
                'for_statement': ['initializer', 'condition', 'update', 'body'],
                'switch_statement': ['condition', 'body'],
            },
            'cpp': {
                'if_statement': ['condition', 'consequence', 'alternative'],
                'while_statement': ['condition', 'body'],
                'for_statement': ['initializer', 'condition', 'update', 'body'],
                'for_range_statement': ['declaration', 'right', 'body'],
            },
            'java': {
                'if_statement': ['condition', 'consequence', 'alternative'],
                'while_statement': ['condition', 'body'],
                'for_statement': ['init', 'condition', 'update', 'body'],
                'enhanced_for_statement': ['type', 'name', 'value', 'body'],
            }
        }
        
        # Build node lookup
        node_by_id = {n['id']: n for n in nodes}
        
        # Find control flow relationships
        for edge in edges:
            source_node = node_by_id.get(edge['source'])
            target_node = node_by_id.get(edge['target'])
            
            if source_node and source_node['original_type'] in cf_patterns.get(lang, {}):
                # This is a control flow parent - mark edge as CONTROL_FLOW
                cfg_edges.append({
                    'source': edge['source'],
                    'target': edge['target'],
                    'type': 'CONTROL_FLOW'
                })
        
        return cfg_edges

    def _add_next_token_edges(self, nodes):
        """
        Add NEXT_TOKEN edges between sequential statement nodes.
        Helps GNN learn execution order.
        """
        next_edges = []
        
        # Filter to statement-level nodes (not deep into expressions)
        statement_nodes = [n for n in nodes if n['type'] in ['DECLARATION', 'PROPERTY']]
        
        # Connect sequential statements
        for i in range(len(statement_nodes) - 1):
            next_edges.append({
                'source': statement_nodes[i]['id'],
                'target': statement_nodes[i + 1]['id'],
                'type': 'NEXT_TOKEN'
            })
        
        return next_edges


    def __len__(self):
        """Return the number of code snippets."""
        return len(self._code)

    def __repr__(self):
        """String representation of the IPAGBuilder."""
        return (f"IPAGBuilder(snippets={len(self._code)}, "
                f"languages={self._lang.nunique()}, "
                f"asts_built={len(self._asts)})")