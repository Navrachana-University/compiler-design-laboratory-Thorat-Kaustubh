import sys
from enum import Enum
import os

###############################################################################
# SPRACHDEFINITION / LANGUAGE DEFINITION
###############################################################################

"""
GRAMMATIK (EBNF) / GRAMMAR (EBNF):

Programm      = { Anweisung } / Program = { Statement }
Anweisung     = Zuweisung | Druck | Wenn | Solange | Block 
Zuweisung     = Bezeichner "=" Ausdruck ";" 
Druck         = "drucke" Ausdruck ";" 
Wenn          = "wenn" "(" Ausdruck ")" Block [ "sonst" Block ] 
Solange       = "solange" "(" Ausdruck ")" Block 
Block         = "{" { Anweisung } "}" 
Ausdruck      = Term { ("+" | "-" | "<" | ">" | "==" | "!=") Term } 
Term          = Faktor { ("*" | "/") Faktor } 
Faktor        = Ganzzahl | Bezeichner | "(" Ausdruck ")" | UnärOp Faktor 
UnärOp        = "-" | "!" 

DATENTYPEN / DATA TYPES:
- Ganzzahlen (z.B. 42) / Integers (e.g., 42)
- Boolesche Werte ("wahr"/"falsch") / Booleans ("true"/"false")
"""

###############################################################################
# PHASE 1: LEXIKALISCHE ANALYSE / LEXICAL ANALYSIS
###############################################################################

class TokenType(Enum):
    # Einzelzeichen-Tokens / Single-character tokens
    LPAREN = '('
    RPAREN = ')'
    LBRACE = '{'
    RBRACE = '}'
    SEMICOLON = ';'
    
    # Operatoren / Operators
    PLUS = '+'
    MINUS = '-'
    STAR = '*'
    SLASH = '/'
    BANG = '!'
    EQUAL = '='
    EQEQ = '=='
    NOTEQ = '!='
    LT = '<'
    LTEQ = '<='
    GT = '>'
    GTEQ = '>='
    
    # Literale / Literals
    IDENTIFIER = 'IDENTIFIER'
    INTEGER = 'INTEGER'
    
    # Schlüsselwörter / Keywords
    IF = 'IF'        # wenn
    ELSE = 'ELSE'    # sonst
    WHILE = 'WHILE'  # solange
    PRINT = 'PRINT'  # drucke
    TRUE = 'TRUE'    # wahr
    FALSE = 'FALSE'  # falsch
    
    EOF = 'EOF'

KEYWORDS = {
    'wenn': TokenType.IF,
    'sonst': TokenType.ELSE,
    'solange': TokenType.WHILE,
    'drucke': TokenType.PRINT,
    'wahr': TokenType.TRUE,
    'falsch': TokenType.FALSE
}

class Token:
    def __init__(self, type, value=None, line=1):
        self.type = type
        self.value = value
        self.line = line
    
    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)}, Line {self.line})"

class Lexer:
    def __init__(self, source):
        self.source = source
        self.tokens = []
        self.start = 0
        self.current = 0
        self.line = 1
        
    def scan_tokens(self):
        print("\n=== Lexikalische Analyse gestartet / Lexical analysis started ===")
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        
        self.tokens.append(Token(TokenType.EOF, None, self.line))
        self.print_tokens()
        return self.tokens
    
    def print_tokens(self):
        print("\nErkannte Tokens / Recognized tokens:")
        for token in self.tokens:
            print(f"Line {token.line}: {token.type.name} ({token.value})")
    
    def scan_token(self):
        c = self.advance()
        
        if c == '(': self.add_token(TokenType.LPAREN)
        elif c == ')': self.add_token(TokenType.RPAREN)
        elif c == '{': self.add_token(TokenType.LBRACE)
        elif c == '}': self.add_token(TokenType.RBRACE)
        elif c == ';': self.add_token(TokenType.SEMICOLON)
        elif c == '+': self.add_token(TokenType.PLUS)
        elif c == '-': self.add_token(TokenType.MINUS)
        elif c == '*': self.add_token(TokenType.STAR)
        elif c == '/': self.add_token(TokenType.SLASH)
        elif c == '!': 
            self.add_token(TokenType.NOTEQ if self.match('=') else TokenType.BANG)
        elif c == '=': 
            self.add_token(TokenType.EQEQ if self.match('=') else TokenType.EQUAL)
        elif c == '<': 
            self.add_token(TokenType.LTEQ if self.match('=') else TokenType.LT)
        elif c == '>': 
            self.add_token(TokenType.GTEQ if self.match('=') else TokenType.GT)
        elif c == ' ' or c == '\r' or c == '\t': pass
        elif c == '\n': self.line += 1
        elif c == '#':
            while self.peek() != '\n' and not self.is_at_end():
                self.advance()
        elif c.isdigit(): self.number()
        elif c.isalpha() or c == '_': self.identifier()
        else:
            self.error(f"Unerwartetes Zeichen / Unexpected character: '{c}'")
    
    def number(self):
        while self.peek().isdigit():
            self.advance()
        self.add_token(TokenType.INTEGER, int(self.source[self.start:self.current]))
    
    def identifier(self):
        while self.peek().isalnum() or self.peek() == '_':
            self.advance()
        text = self.source[self.start:self.current]
        token_type = KEYWORDS.get(text, TokenType.IDENTIFIER)
        self.add_token(token_type, text)
    
    def add_token(self, type, value=None):
        self.tokens.append(Token(type, value, self.line))
    
    def advance(self):
        self.current += 1
        return self.source[self.current - 1]
    
    def peek(self):
        return '\0' if self.is_at_end() else self.source[self.current]
    
    def match(self, expected):
        if self.is_at_end() or self.source[self.current] != expected:
            return False
        self.current += 1
        return True
    
    def is_at_end(self):
        return self.current >= len(self.source)
    
    def error(self, message):
        raise Exception(f"[Line {self.line}] {message}")

###############################################################################
# PHASE 2: SYNTAXANALYSE / SYNTAX ANALYSIS
###############################################################################

class ASTNode: pass

class Program(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class Block(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class Assign(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value

class If(ASTNode):
    def __init__(self, condition, then_branch, else_branch=None):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

class While(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class Print(ASTNode):
    def __init__(self, expression):
        self.expression = expression

class BinaryOp(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class UnaryOp(ASTNode):
    def __init__(self, op, right):
        self.op = op
        self.right = right

class Literal(ASTNode):
    def __init__(self, value):
        self.value = value

class Variable(ASTNode):
    def __init__(self, name):
        self.name = name

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0
        print("\n=== Syntaxanalyse gestartet / Syntax analysis started ===")
    
    def parse(self):
        statements = []
        while not self.is_at_end():
            statements.append(self.statement())
        return Program(statements)
    
    def statement(self):
        if self.match(TokenType.PRINT): return self.print_statement()
        if self.match(TokenType.IF): return self.if_statement()
        if self.match(TokenType.WHILE): return self.while_statement()
        if self.match(TokenType.LBRACE): return Block(self.block())
        return self.expression_statement()
    
    def print_statement(self):
        value = self.expression()
        self.consume(TokenType.SEMICOLON, "Erwarte ';' nach Wert / Expect ';' after value")
        return Print(value)
    
    def if_statement(self):
        self.consume(TokenType.LPAREN, "Erwarte '(' nach 'wenn' / Expect '(' after 'if'")
        condition = self.expression()
        self.consume(TokenType.RPAREN, "Erwarte ')' nach Bedingung / Expect ')' after condition")
        
        then_branch = self.statement()
        else_branch = None
        if self.match(TokenType.ELSE):
            else_branch = self.statement()
        
        return If(condition, then_branch, else_branch)
    
    def while_statement(self):
        self.consume(TokenType.LPAREN, "Erwarte '(' nach 'solange' / Expect '(' after 'while'")
        condition = self.expression()
        self.consume(TokenType.RPAREN, "Erwarte ')' nach Bedingung / Expect ')' after condition")
        body = self.statement()
        return While(condition, body)
    
    def expression_statement(self):
        expr = self.expression()
        self.consume(TokenType.SEMICOLON, "Erwarte ';' nach Ausdruck / Expect ';' after expression")
        return expr
    
    def block(self):
        statements = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            statements.append(self.statement())
        self.consume(TokenType.RBRACE, "Erwarte '}' nach Block / Expect '}' after block")
        return statements
    
    def expression(self):
        return self.assignment()
    
    def assignment(self):
        expr = self.equality()
        
        if self.match(TokenType.EQUAL):
            equals = self.previous()
            value = self.assignment()
            
            if isinstance(expr, Variable):
                return Assign(expr.name, value)
            
            self.error(equals, "Ungültige Zuweisung / Invalid assignment target")
        
        return expr
    
    def equality(self):
        expr = self.comparison()
        
        while self.match(TokenType.NOTEQ, TokenType.EQEQ):
            operator = self.previous()
            right = self.comparison()
            expr = BinaryOp(expr, operator.type, right)
        
        return expr
    
    def comparison(self):
        expr = self.term()
        
        while self.match(TokenType.LT, TokenType.LTEQ, TokenType.GT, TokenType.GTEQ):
            operator = self.previous()
            right = self.term()
            expr = BinaryOp(expr, operator.type, right)
        
        return expr
    
    def term(self):
        expr = self.factor()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.previous()
            right = self.factor()
            expr = BinaryOp(expr, operator.type, right)
        
        return expr
    
    def factor(self):
        expr = self.unary()
        
        while self.match(TokenType.STAR, TokenType.SLASH):
            operator = self.previous()
            right = self.unary()
            expr = BinaryOp(expr, operator.type, right)
        
        return expr
    
    def unary(self):
        if self.match(TokenType.BANG, TokenType.MINUS):
            operator = self.previous()
            right = self.unary()
            return UnaryOp(operator.type, right)
        
        return self.primary()
    
    def primary(self):
        if self.match(TokenType.FALSE): return Literal(False)
        if self.match(TokenType.TRUE): return Literal(True)
        if self.match(TokenType.INTEGER): return Literal(self.previous().value)
        if self.match(TokenType.IDENTIFIER): return Variable(self.previous().value)
        if self.match(TokenType.LPAREN):
            expr = self.expression()
            self.consume(TokenType.RPAREN, "Erwarte ')' nach Ausdruck / Expect ')' after expression")
            return expr
        
        self.error(self.peek(), "Erwarte Ausdruck / Expect expression")
    
    def match(self, *types):
        for type in types:
            if self.check(type):
                self.advance()
                return True
        return False
    
    def check(self, type):
        if self.is_at_end():
            return False
        return self.peek().type == type
    
    def advance(self):
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def is_at_end(self):
        return self.peek().type == TokenType.EOF
    
    def peek(self):
        return self.tokens[self.current]
    
    def previous(self):
        return self.tokens[self.current - 1]
    
    def consume(self, type, message):
        if self.check(type):
            return self.advance()
        
        self.error(self.peek(), message)
    
    def error(self, token, message):
        raise Exception(f"[Line {token.line}] {message}")
    
    def synchronize(self):
        self.advance()
        
        while not self.is_at_end():
            if self.previous().type == TokenType.SEMICOLON:
                return
            
            if self.peek().type in [TokenType.IF, TokenType.WHILE, TokenType.PRINT]:
                return
            
            self.advance()

###############################################################################
# PHASE 3: SEMANTISCHE ANALYSE / SEMANTIC ANALYSIS
###############################################################################

class SemanticAnalyzer:
    def __init__(self):
        self.scopes = [{}]
        print("\n=== Semantische Analyse gestartet / Semantic analysis started ===")
    
    def analyze(self, node):
        if isinstance(node, Program):
            for statement in node.statements:
                self.analyze(statement)
        elif isinstance(node, Block):
            self.enter_scope()
            for statement in node.statements:
                self.analyze(statement)
            self.exit_scope()
        elif isinstance(node, Assign):
            # Implizite Variablendeklaration / Implicit variable declaration
            if node.name not in self.scopes[-1]:
                self.scopes[-1][node.name] = True
            self.analyze(node.value)
        elif isinstance(node, If):
            self.analyze(node.condition)
            self.analyze(node.then_branch)
            if node.else_branch:
                self.analyze(node.else_branch)
        elif isinstance(node, While):
            self.analyze(node.condition)
            self.analyze(node.body)
        elif isinstance(node, Print):
            self.analyze(node.expression)
        elif isinstance(node, BinaryOp):
            self.analyze(node.left)
            self.analyze(node.right)
        elif isinstance(node, UnaryOp):
            self.analyze(node.right)
        elif isinstance(node, Variable):
            if not any(node.name in scope for scope in self.scopes):
                raise Exception(f"Undefinierte Variable / Undefined variable: '{node.name}'")
        elif isinstance(node, Literal):
            pass
    
    def enter_scope(self):
        self.scopes.append({})
    
    def exit_scope(self):
        self.scopes.pop()

###############################################################################
# PHASE 4: ZWISCHENCODE-GENERIERUNG / INTERMEDIATE CODE GENERATION
###############################################################################

class IRGenerator:
    def __init__(self):
        self.code = []
        self.temp_count = 0
        self.label_count = 0
        print("\n=== Zwischencode-Generierung gestartet / Intermediate code generation started ===")
    
    def generate(self, node):
        if isinstance(node, Program):
            for statement in node.statements:
                self.generate(statement)
        elif isinstance(node, Block):
            for statement in node.statements:
                self.generate(statement)
        elif isinstance(node, Assign):
            temp = self.generate(node.value)
            self.emit(f"{node.name} = {temp}")
        elif isinstance(node, If):
            condition_temp = self.generate(node.condition)
            else_label = self.new_label()
            end_label = self.new_label()
            
            self.emit(f"if not {condition_temp} goto {else_label}")
            self.generate(node.then_branch)
            
            if node.else_branch:
                self.emit(f"goto {end_label}")
                self.emit(f"{else_label}:")
                self.generate(node.else_branch)
                self.emit(f"{end_label}:")
            else:
                self.emit(f"{else_label}:")
        elif isinstance(node, While):
            start_label = self.new_label()
            end_label = self.new_label()
            
            self.emit(f"{start_label}:")
            condition_temp = self.generate(node.condition)
            self.emit(f"if not {condition_temp} goto {end_label}")
            self.generate(node.body)
            self.emit(f"goto {start_label}")
            self.emit(f"{end_label}:")
        elif isinstance(node, Print):
            expr_temp = self.generate(node.expression)
            self.emit(f"print {expr_temp}")
        elif isinstance(node, BinaryOp):
            left_temp = self.generate(node.left)
            right_temp = self.generate(node.right)
            result_temp = self.new_temp()
            
            op_map = {
                TokenType.PLUS: '+',
                TokenType.MINUS: '-',
                TokenType.STAR: '*',
                TokenType.SLASH: '/',
                TokenType.EQEQ: '==',
                TokenType.NOTEQ: '!=',
                TokenType.LT: '<',
                TokenType.LTEQ: '<=',
                TokenType.GT: '>',
                TokenType.GTEQ: '>='
            }
            
            self.emit(f"{result_temp} = {left_temp} {op_map[node.op]} {right_temp}")
            return result_temp
        elif isinstance(node, UnaryOp):
            operand_temp = self.generate(node.right)
            result_temp = self.new_temp()
            
            op_map = {
                TokenType.MINUS: '-',
                TokenType.BANG: '!'
            }
            
            self.emit(f"{result_temp} = {op_map[node.op]}{operand_temp}")
            return result_temp
        elif isinstance(node, Variable):
            return node.name
        elif isinstance(node, Literal):
            return str(int(node.value) if isinstance(node.value, bool) else node.value)
        
        return None
    
    def new_temp(self):
        temp = f"t{self.temp_count}"
        self.temp_count += 1
        return temp
    
    def new_label(self):
        label = f"L{self.label_count}"
        self.label_count += 1
        return label
    
    def emit(self, instruction):
        self.code.append(instruction)
    
    def get_code(self):
        return self.code

###############################################################################
# PHASE 5: OPTIMIERUNG / OPTIMIZATION
###############################################################################

class Optimizer:
    def optimize(self, code):
        print("\n=== Code-Optimierung gestartet / Code optimization started ===")
        optimized = []
        i = 0
        while i < len(code):
            current = code[i]
            
            # Konstantenfaltung / Constant folding
            if '=' in current and any(op in current for op in ['+', '-', '*', '/']):
                parts = current.split('=')
                var = parts[0].strip()
                expr = parts[1].strip()
                
                try:
                    result = str(eval(expr))
                    optimized.append(f"{var} = {result}")
                    i += 1
                    continue
                except:
                    pass
            
            # Redundante Zuweisungen entfernen / Remove redundant assignments
            if '=' in current:
                parts = current.split('=')
                var = parts[0].strip()
                value = parts[1].strip()
                if var == value:
                    i += 1
                    continue
            
            optimized.append(current)
            i += 1
        
        print("\nOptimierter Code / Optimized code:")
        for instr in optimized:
            print(instr)
        
        return optimized

###############################################################################
# COMPILER DRIVER
###############################################################################

class Compiler:
    def __init__(self):
        self.lexer = None
        self.parser = None
        self.semantic_analyzer = SemanticAnalyzer()
        self.ir_generator = IRGenerator()
        self.optimizer = Optimizer()
    
    def compile(self, source):
        try:
            # 1. Lexikalische Analyse / Lexical Analysis
            self.lexer = Lexer(source)
            tokens = self.lexer.scan_tokens()
            
            # 2. Syntaxanalyse / Syntax Analysis
            self.parser = Parser(tokens)
            ast = self.parser.parse()
            
            # 3. Semantische Analyse / Semantic Analysis
            self.semantic_analyzer.analyze(ast)
            
            # 4. Zwischencode-Generierung / Intermediate Code Generation
            self.ir_generator.generate(ast)
            intermediate_code = self.ir_generator.get_code()
            
            # 5. Optimierung / Optimization
            optimized_code = self.optimizer.optimize(intermediate_code)
            
            return optimized_code
        except Exception as e:
            print(f"\nFehler während der Kompilierung / Compilation error: {str(e)}", file=sys.stderr)
            return None

###############################################################################
# MAIN PROGRAM
###############################################################################

def read_source_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            source = file.read()
            print(f"\nQuellcode / Source code from {file_path}:")
            print(source)
            return source
    except IOError as e:
        print(f"\nDateifehler / File error: {str(e)}", file=sys.stderr)
        return None

def write_output_file(file_path, content):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(content))
        print(f"\nAusgabe gespeichert in / Output saved to: {file_path}")
    except IOError as e:
        print(f"\nDateifehler / File error: {str(e)}", file=sys.stderr)

def main():
    if len(sys.argv) != 2:
        print("\nVerwendung: python compiler.py <Eingabedatei>", file=sys.stderr)
        print("Usage: python compiler.py <input_file>", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"\nFehler: Datei nicht gefunden / Error: File not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    source = read_source_file(input_file)
    if source is None:
        sys.exit(1)
    
    compiler = Compiler()
    optimized_code = compiler.compile(source)
    
    if optimized_code is not None:
        output_file = os.path.splitext(input_file)[0] + ".ir"
        write_output_file(output_file, optimized_code)

if __name__ == "__main__":
    main()