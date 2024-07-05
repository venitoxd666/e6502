from typing import *

import io, sys, os
import string
import struct
import argparse
from enum import IntEnum
from copy import deepcopy

InputBuffer = io.StringIO # even though files may not return this type, the functionality is basically the same
OutputBuffer = io.BytesIO # same here, but remember, always write bytes

Ignored =  Any

class InvalidSyntaxError(Exception):
    pass

class TokenType(IntEnum):
    SYMBOL = 0          # done //
    DIRECTIVE = 1       # done
    IDENTIFIER = 2      # done //
    INTEGER_LITERAL = 3 # done //
    STRING_LITERAL = 4  # not now (not done) (TODO: add process_string) when `"` is found //
    EOL = 5             # done //
    EOF = 6             # done //

class MemoryAccess(IntEnum):
    INMEDIATE = 1       # inmediate value
    POINTER = 2         # memory pointer
    OFFSET = 3          # memory offset from PC or SP (if stack operations)
    DIRECT_PAGE = 4     # this makes no sense to store it in a register (does it?)
    DETECT = 5          # shall god bless you

SYMBOLS = set('¡¿?\'\"\\&%$#@·!/(){}<>[]~|*+-,:;')
DIRECTIVES = {
    "section",
    "func",
    "label",
    'data'
}

MEMORY_ACCESS_NAMES = {
    'ptr':MemoryAccess.POINTER,
    'inm':MemoryAccess.INMEDIATE,
    'off':MemoryAccess.OFFSET,
    'dp':MemoryAccess.DIRECT_PAGE,
    'pointer':MemoryAccess.POINTER,
    'inmediate':MemoryAccess.INMEDIATE,
    'offset':MemoryAccess.OFFSET,
    'direct_page':MemoryAccess.DIRECT_PAGE
}

MEMORY_ACCESS_SYMBOLS = {
    '*':MemoryAccess.POINTER,
    '#':MemoryAccess.INMEDIATE,
    ':':MemoryAccess.OFFSET,
    '$':MemoryAccess.DIRECT_PAGE
}

SIZES_NAMES = {
    'byte':   1,
    'db':     1,
    'short':  2,
    'word':   2,
    'ds':     2,
    'dw':     2,
    'int':    4,
    'di':     4,
    'long':   8,
    'dl':     8
}

SIZE_DESCR_SYMB = {
    '-':1,
    '^':2,
    '!':4,
    '¡':8
}

class Auditable(object):
    # these are common
    name:str
    audit:int

class Position(object):
    cursor:int
    line:int
    inline:int

    def __init__(self):
        self.cursor = self.line = self.inline = 0

    def advance(self, char:str) -> None:
        if not char:
            return
        self.cursor += 1
        self.inline += 1
        if char == '\n':
            self.inline = 0
            self.line += 1

    def copy(self, _from = None) -> "Position":
        if _from:
            self.cursor = _from.cursor
            self.inline = _from.inline
            self.line = _from.line
            return self

        return Position().copy(_from = self)

    def repr(self):
        return f"[Position, line = {self.line}, position = {self.inline}]"

class Token:
    type:TokenType
    value: Optional[Any]
    pos: Optional[Position]

    def __init__(self, type: TokenType, value: Optional[str] = None, position: Optional[Position] = None)  -> None:
        self.type = type
        self.value = value
        self.pos = None
        if position:
            self.pos = position.copy()

    def _repr_with_pos_and_value(self):
        return f"[Token type={self.tp()}, value = \"{self.value}\", line = {self.pos.line}, position = {self.pos.inline}]"

    def _repr_with_pos_no_val(self):
        return f"[Token type={self.tp()}, line = {self.pos.line}, position = {self.pos.inline}]"

    def pos_repr(self):
        return self.pos.repr()

    repr_pos = pos_repr

    def _repr_with_pos(self):
        if self.value:
            return self._repr_with_pos_and_value()
        return self._repr_with_pos_no_val()

    def _repr_value(self):
        return f"[Token type={self.tp()}, value = \"{self.value}\"]"

    def _repr_no_value(self):
        return f"[Token type={self.tp()}]"

    def tp(self):
        return self.type.name

    def _repr_no_pos(self):
        if self.value:
            return self._repr_value()
        return self._repr_no_value()

    def __repr__(self):
        if self.pos:
            return self._repr_with_pos()
        return self._repr_no_pos()

    __str__ = __repr__

class SpecialList(list):
    def __init__(self, callback):
        self._callback = callback
        super().__init__()

    def append(self, val):
        self._callback(val)
        super().append(val)

class Tokenizer(object):

    stdin: InputBuffer
    tokens: List[Token]
    position: Position

    def __init__(self, stdin:InputBuffer) -> None:
        self.stdin = stdin
        self.tokens = SpecialList(self.on_tok_append)
        self.position = Position()
        self.curr =  ""

    def on_tok_append(self, token:Token) -> Ignored:
        pass

    def advance(self):
        new_char = self.stdin.read(1)
        self.position.advance(new_char)
        self.curr = new_char
        return self.curr

    def process_digit(self) -> None:
        buffer = self.curr
        n_digits = 10
        digits = '0123456789abcdef'
        if buffer == '0':
            _next_char = self.advance()
            if   _next_char.lower() == 'x': n_digits = 16;self.advance()
            elif _next_char.lower() == 'b': n_digits = 2; self.advance()
            elif _next_char.lower() == 'c': n_digits = 8; self.advance()
            elif  not _next_char.isdigit():
                self.tokens.append(Token(TokenType.INTEGER_LITERAL, value=int(buffer), position = self.position))
                return
        else:
            self.advance() # consume already in-buffer char
        _digits_section  = digits[:n_digits]
        _cond = lambda :self.curr.lower() in _digits_section
        while(_cond() and not self.eof()):
            buffer += self.curr
            self.advance()

        self.tokens.append(Token(TokenType.INTEGER_LITERAL, value=int(buffer, base = n_digits), position = self.position))

    def _cond_and_not_eof(self, cond:Callable) -> Callable:
        return lambda : cond() and not(self.eof())

    def eof(self):
        return self.curr == '' and self.stdin.tell()

    def eol(self):
        return self.curr == '\n'

    def invalid_syntax(self, msg:str):
        raise InvalidSyntaxError(msg +  f'at line {self.position.line}, pos {self.position.inline}')

    def process_identifier(self,):
        buffer = ""
        while(not(self.curr in SYMBOLS) and not self.eof() and not self.eol() and not self.curr==' '):
            buffer += self.curr
            self.advance()

        self.tokens.append(Token(TokenType.IDENTIFIER, value = buffer, position = self.position))

        return

    def process_long_comment(self):
        buff = self.advance()
        buff += self.advance()
        while (not self.eof() and buff != '*;'):
            buff = buff[1] + self.advance()

        if self.eof():
            self.invalid_syntax("unexpected EOF while scanning long comment")

        return

    def process_comment(self):
        self.advance()
        if self.curr == '*':
            self.process_long_comment()
            return

        while(not self.eol() and not self.eof()):
            self.advance()

        return

    def tokenize(self):
        self.advance()
        while(self.curr):
            #  todo tokenizing stuff
            if(self.curr.isdigit()):
                self.process_digit()
                continue
            if (self.eol()):
                self.tokens.append(Token(TokenType.EOL, position = self.position))
                self.advance()
                continue
            if(self.curr == ';'):
                self.process_comment()
                continue
            if(self.curr in SYMBOLS):
                self.tokens.append(Token(TokenType.SYMBOL, value = self.curr, position = self.position))
                self.advance()
                continue
            if(self.curr == ' ' or self.curr == '\t'):
                self.advance()
                continue
            if(self.curr == '.'):
                self.tokens.append(Token(TokenType.DIRECTIVE, position = self.position))
                self.advance()
                continue
            else:
                self.process_identifier()
                continue
            self.advance()

        self.tokens.append(Token(TokenType.EOF, position = self.position))
        return self.tokens

class Operand:
    def __init__(self, memory_access, type, value, size):
        self.memory_access = memory_access
        self.type = type
        self.value = value
        self.size = size # -1 if unknown

    def __str__(self):
        return f"Operand(memory_access={self.memory_access}, value={self.value}, size = {self.size}, type = {self.type.name})"

    __repr__ = __str__

class Opcode:
    def __init__(self, opcode):
        assert len(opcode)
        self.opcode = opcode
        self.composed = bool(isinstance(opcode, list) and (len(opcode) - 1))
        if isinstance(opcode, list) and not (len(opcode) - 1):
            self.opcode = self.opcode[0]

    def __len__(self):
        if not self.composed:
            return 1
        return len(self.opcode)

    @property
    def name(self):
        if self.composed:
            return ' '.join(self.opcode)
        return self.opcode

class Instruction(object):
    opcode: Opcode
    operands:List[Operand]
    code: Optional["RawCode"]
    address: Optional[int]

    def __init__(self, opcode, operands):
        self.opcode = opcode
        self.operands = operands
        self.code = None
        self.address = None # address = None | address = FUTURE

    def __repr__(self):
        return f"<Instruction opcode = {self.opcode.name}, operands = {self.operands}>"

    __str__ = __repr__


class Cursor(object):

    def __init__(self):
        self.section = None
        self.function = None
        self.label = None

    def on_function(self, funcname:str,**meta):
        if not self.section:
            raise ValueError(f"must be in a section to write a function in")
        self.function = self.section.create_function(funcname,**meta)
        return

    def on_label(self, label:str):
        if not self.function:
            raise ValueError(f"must be in a function to write a label in")

        self.function.create_label(label)
        self.label = label

    def is_in_not_label(self):
        return f'section {self.section.name if self.section else "<no-section>"}, function {self.function.name if self.function else "<no-function>"}'

    def is_in(self,label=True):
        if not label:
            return self.is_in_not_label()
        return f'section {self.section.name if self.section else "<no-section>"}, function {self.function.name if self.function else "<no-function>"}, label {self.label if self.label else "<no-label>"}'

class Function(Auditable):
    def __init__(self, name:str,section:"Section", audit=-1, **meta):
        self.name = name
        self.instr = []
        self.labels = {}
        self.meta = meta
        self.audit = audit
        self.section = section
        self.org = meta.get('org', -1)
        self.dumped = False # gonna be reused for Function dumping information

    def _get_symbol(self, name:str) -> Optional["Symbol"]:
        if not name in self.labels:
            return None

        return Symbol(SymbolTypes.LABEL, name, self.section)

    def create_label(self, name):
        if name in self.labels:
            return
        self.labels[name] = len(self.instr)

class DataDefinition(Auditable):
    def __init__(self, size, name, value, audit = -1):
        self.size  = size
        self.value = value
        self.name = name
        self.audit = audit

class Section(object):

    audit:List[Auditable]
    decl:Dict[str, DataDefinition]
    functions:Dict[str, Function]
    org:int
    size:int
    fillbyte:int
    fill:int
    glob:int

    def __init__(self, name:str, org:int = -1, fillbyte = 0, fill = -1, glob = False):
        self.name = name
        self.functions = {}
        self.decl      = {}
        self.audit     = []
        if not isinstance(org, int):
            raise ValueError(f"bad `org` parameter given to section {name}; expected int literal")
        self.org     = org # -1  <-> not set
        self.size    = -1  # -1  <-> not computed
        if fillbyte & 0xFF != fillbyte:
            raise ValueError(f"bad `fillbyte` parameter given to section {name}; expected a byte")
        self.fillbyte = fillbyte
        self.fill = fill
        self.glob = glob

    def data_declaration(self, name, decl):
        self.decl[name] = decl
        decl.audit = len(self.audit)
        self.audit.append(self.decl[name]) # len(self.audit) is current index.

    def create_function(self, name,*_, at=-1,**meta):
        if name in self.functions:
            raise ValueError(f"function {name} instanciated twice in the same section")
        self.functions[name] = Function(name,self,audit = len(self.audit) if at == -1 else at, **meta)
        if at < 0:
            self.audit.append(self.functions[name])
        else:
            self.audit.insert(at, self.functions[name])

        return self.functions[name]

class SymbolType(IntEnum):
    LINKED           = 0
    UNDEFINED        = 1
    DATA_DECLARATION = 2

class SymbolObject(IntEnum):
    SECTION          = 0
    FUNCTION         = 1
    INSTRUCTION      = 2
    DATA_DECLARATION = 3
    UNKNOWN          = 4

SymbolObjectType = Optional[Union[Instruction, Function, Section, DataDefinition]]

class Symbol:
    linked: Optional[Union["Symbol", int]] # int just in case it's a FUNCTION Symbol, as the integer represents the instruction's index.
    type  : SymbolType

    name:   str
    obj:    SymbolObjectType
    object: SymbolObject
    audit:  int

    def __init__(self, type:SymbolType, name:str, object:SymbolObject, audit:int, linked:Optional["Symbol"] = None, obj:SymbolObjectType = None,msg:str= None):
        self.type = type
        self.name = name
        if isinstance(linked, int) and linked < 0:
            raise ValueError(msg or f"Invalid definition, registered linked symbol to nothing")
        self.linked = linked
        self.obj = obj
        self.object = object
        self.address = -1


class OutputFile(object):

    symb_table:Dict[str, Symbol]

    def __init__(self):
        self.sections = {}
        self.curr_func = None
        self.cursor = Cursor()
        self.symb_table = {}
        self.linker = Linker(self)

    def in_section(self, section_name:str, meta:Optional[Dict[str, Any]]=None):
        if meta is None: meta = {}
        if not section_name in self.sections:
            self.sections[section_name] = Section(section_name, **meta)
        self.cursor.section = self.sections[section_name]

    def in_function(self, function_name:str,**meta):
        self.cursor.on_function(function_name,**meta)

    def in_label(self, label_name:str):
        self.cursor.on_label(label_name)

    def whereami(self, label = True):
        return self.cursor.is_in(label = label)

    def write(self, instruction:Instruction) -> None:
        if not isinstance(instruction, Instruction):
            raise ValueError("?")
        if not self.cursor.function or not self.cursor.section:
            raise ValueError("Cannot write instruction into a function if no function is being defined")
        self.cursor.function.instr.append(instruction)




class Architecture(object):
    def __init__(self, name, registers, opcodes, opc_enum):
        self.name      = name
        self.registers = registers
        self.opcodes   = opcodes
        self.opc_enum  = opc_enum

    def registered_name(self, name):
        return (name in self.opcodes) or (name in self.registers)

class Linker(object):

    def __init__(self, ofile):
        self.ofile = ofile
        self.arch = None

    def conf_arch(self, arch:Architecture):
        if not isinstance(arch, Architecture):
            raise ValueError(...)

        self.arch = arch

    def in_table(self, symbol:str) -> Ignored:
        if symbol in self.table:
            raise ValueError(f"Cannot register symbol {symbol} because it's defined twice")

    def register_symbols(self):
        for secname, section in self.ofile.sections.items():
            if self.arch and self.arch.registered_name(secname):
                raise ValueError(f"section with protected name: {secname}")
            self.in_table(secname)
            self.table[secname] = Symbol(SymbolType.LINKED, secname, SymbolObject.SECTION, audit=-1, linked = section.audit[0].name if len(section.audit) else -1,
                obj = section,msg = 'Section empty')

            for funcname, function in section.functions.items():
                if self.arch and self.arch.registered_name(funcname):
                    raise ValueError(f"Function with protected name: {funcname}")

                self.in_table(funcname)
                self.table[funcname] = Symbol(SymbolType.LINKED, funcname, SymbolObject.FUNCTION, audit = function.audit, linked = 0, obj = function)

            for cname, cdecl in section.decl.items():
                if self.arch and self.arch.registered_name(cname):
                    raise ValueError(f"data-declaration with protected name: {cname}")

                self.in_table(cname)
                self.table[cname] = Symbol(SymbolType.DATA_DECLARATION, cname, SymbolObject.DATA_DECLARATION, audit = cdecl.audit, linked = 0, obj = cdecl)

    @property
    def table(self):
        return self.ofile.symb_table

    def resolve_symbol(self, symb):
        pass

    def link(self):
        if not self.arch:
            raise ValueError(f"cannot link before specifying architecture.")
        pass

class Parser(object):

    ofile: OutputFile

    def __init__(self, tokens:List[Token]):
        self.tokens = tokens
        self.ofile = OutputFile()
        self.tok_ptr = 0
        self.curr_tok = self.tokens[0]

    def advance(self):
        self.tok_ptr += 1
        if self.tok_ptr == len(self.tokens):
            return
        self.curr_tok = self.tokens[self.tok_ptr]
        return self.curr_tok

    def peek(self):
        if self.tok_ptr + 1 == len(self.tokens):
            return None
        return self.tokens[self.tok_ptr + 1]

    def expect(self, tok_type, curr = False, rt_peek = None):
        rt_peek =  rt_peek if rt_peek is not None else 0
        peek = self.curr_tok if curr else self.peek()
        if not peek:
            raise ValueError(f"Expected token of type(s) {tok_type} after {repr(self.curr_tok)} at {self.curr_tok.pos_repr()}, found EOT (End Of Tokens)")
        if not isinstance(tok_type, (list, tuple)):
            rt_peek = 1
            tok_type = [tok_type]

        if not peek.type in tok_type:
            raise ValueError(f"Expected token of type(s) {tok_type} after {repr(self.curr_tok)} at {self.curr_tok.pos_repr()}, found {peek.tp()}")

        if rt_peek:
            return peek
        return tok_type.index(peek.type)

    def unexpected(self, tok_type):
        peek = self.peek()
        if not peek:
            raise ValueError(f"Unexpected EOT (End of tokens) found at {self.curr_tok.pos_repr()}")
        if not isinstance(tok_type, (list, tuple)):
            tok_type = [tok_type]

        if peek.type in tok_type:
            raise ValueError(f"Expected token of any type but {tok_type} after {repr(self.curr_tok)} at {self.curr_tok.pos_repr()}, found {peek.tp()}")
        return peek.type

    def eof(self):
        return self.curr_tok.type == TokenType.EOF

    def on_section_directive(self):
        meta = {}
        og_pos = self.curr_tok.pos.copy()
        self.expect(TokenType.IDENTIFIER)
        self.advance()
        sec_name = self.curr_tok.value
        if self.expect([TokenType.SYMBOL,TokenType.DIRECTIVE]):
            # print("parsing directives")
            self.advance() # is directive
            meta = self.on_function_read_meta() # we can reuse this code
            self.expect(TokenType.SYMBOL, curr = True)
            # print(self.curr_tok)
        else:
            # print("not parsing directives!")
            self.advance()
        if not self.curr_tok.value == ':':
            raise ValueError(f'Expected \":\" after section declaration of {sec_name} at {og_pos.repr()}')
        self.expect(TokenType.EOL)
        # print("reached in_section() call")
        self.ofile.in_section(sec_name, meta)
        self.advance()
        return

    def on_function_read_meta(self):
        og_pos = self.curr_tok.pos.copy()
        meta = {}
        while(self.curr_tok.type == TokenType.DIRECTIVE):
            self.expect(TokenType.IDENTIFIER)
            self.advance()

            if self.curr_tok.value == 'attr':
                # .func name .attr <any-identifier> <any-value>
                self.expect(TokenType.IDENTIFIER)
                attrname = self.advance().value
                # make sure the next token is a valid value, not some random crap (or worse, EOF)
                _ = self.unexpected([TokenType.SYMBOL, TokenType.DIRECTIVE,TokenType.EOL, TokenType.EOF])
                # this just uses the tokenizer heartwarming work of decoding the information
                meta[attrname] = self.advance().value
                self.advance() # next token
                continue

            if self.curr_tok.value == 'org':
                # .func name .org <address>
                self.expect(TokenType.INTEGER_LITERAL)
                address = self.advance().value
                self.unexpected([TokenType.IDENTIFIER, TokenType.INTEGER_LITERAL, TokenType.STRING_LITERAL, TokenType.EOF, TokenType.EOL])
                meta['org'] = address
                self.advance()
                continue

            if self.curr_tok.value == 'external':
                # .external
                meta['external'] = 1
                self.advance()
                continue
            if self.curr_tok.value == 'glob':
                # export
                meta['glob'] = 1
                self.advance()
                continue
            ...

            raise ValueError(f"Unexpected function attribute modificator: {self.curr_tok.value}, cannot be recognised")

        return meta


    def on_function_directive(self):
        og_pos = self.curr_tok.pos.copy()
        self.expect(TokenType.IDENTIFIER)
        self.advance()
        meta = {}
        func_name = self.curr_tok.value

        if self.expect([TokenType.SYMBOL, TokenType.DIRECTIVE]): # is directive
            self.advance()
            # read function metadata
            meta = self.on_function_read_meta()
            self.expect(TokenType.SYMBOL, curr = True)
        else:
            self.advance()
        if self.curr_tok.value != ':':
            raise ValueError(f"Expected \":\" after function declaration end at {og_pos.repr()}")

        self.expect(TokenType.EOL)
        self.advance()

        self.ofile.in_function(func_name, **meta)


    def on_label_directive(self):
        og_pos = self.curr_tok.pos.copy()
        self.expect(TokenType.IDENTIFIER)
        self.advance()

        if self.expect(TokenType.SYMBOL).value != ':':
            raise ValueError(f"Expected \":\" after label declaration end at {og_pos.repr()} of label {self.curr_tok.name} at {self.ofile.whereami(label = False)}")
        label_name = self.curr_tok.value
        self.advance()
        self.expect(TokenType.EOL)
        self.advance()

        self.ofile.in_label(label_name)

    def on_easy_data_directive(self, bigger = False):
        if self.curr_tok.type == TokenType.EOL:
            self.advance()
            return
        if (self.curr_tok.type == TokenType.SYMBOL and self.curr_tok.value == '}' and bigger):
            return
        self.expect(TokenType.IDENTIFIER, curr = True)
        decl_name = self.curr_tok.value
        if self.expect(TokenType.SYMBOL).value != ':':
            raise ValueError(f"Expected \":\" after data declaration at {self.curr_tok.pos_repr()}")

        self.advance()
        if self.expect(TokenType.SYMBOL).value != '(':
            raise ValueError(f"Expected \"(\" as mark of required size descriptor for data declaration at {self.curr_tok.pos_repr()}")

        self.advance()
        size = self.parse_keyed_size_mod()

        self.expect([TokenType.INTEGER_LITERAL, TokenType.IDENTIFIER,  TokenType.STRING_LITERAL, TokenType.SYMBOL], curr = True)

        if self.curr_tok.type == TokenType.SYMBOL:
            if self.curr_tok.value != '%':
                raise ValueError(f"Expected symbol \"%\" at the start of data macro expansion at {self.curr_tok.pos_repr()}")

            if self.expect(TokenType.SYMBOL).value != '(':
                raise ValueError(f"Expected symbol \"(\" at the start of data macro expansion at {self.curr_tok.pos_repr()}")

            self.advance()
            self.advance()

            val = self.evaluate_operand_macros_expansion()
            self.expect([TokenType.SYMBOL], curr = True)
            if self.curr_tok.value != ')':
                raise ValueError(f"Expected symbol \")\" at the end of data macro expansion at {self.curr_tok.pos_repr()}")
        else:
            val = self.curr_tok.value
        # print(self.curr_tok)
        self.expect([TokenType.EOL, TokenType.SYMBOL,TokenType.EOF] if bigger else [TokenType.EOL,TokenType.EOF])
        self.advance()
        self.ofile.cursor.section.data_declaration(decl_name, DataDefinition(size, decl_name, val))

    def on_data_directive(self):
        if self.peek().type == TokenType.IDENTIFIER:
            self.advance()
            return self.on_easy_data_directive()

        if self.expect(TokenType.SYMBOL).value != '{':
            raise ValueError(f"Expected \"{'{'}\" at the start of expanded data declaration at {self.curr_tok.pos_repr()}")
        self.advance() # curr = {
        while self.peek().type == TokenType.EOL:
            self.advance()

        while(True):
            self.on_easy_data_directive(bigger = True)
            if(self.curr_tok.type==TokenType.EOL):
                self.advance()
                continue
            elif(self.curr_tok.type==TokenType.SYMBOL and self.curr_tok.value == '}'):
                break
            elif (self.curr_tok.type==TokenType.EOF):
                raise ValueError(f"Unexpected EOF while scanning data segment of section {self.cursor.section.name} at {self.curr_tok.pos_repr()}")

        self.advance()

    def directive(self):
        self.expect(TokenType.IDENTIFIER)
        self.advance()

        if self.curr_tok.value == 'section':
            self.on_section_directive()
            return
        if self.curr_tok.value == 'func':
            self.on_function_directive()
            return
        if self.curr_tok.value == 'label':
            self.on_label_directive()
            return
        if self.curr_tok.value == 'data':
            self.on_data_directive()
            return
        raise ValueError(f"Expected a valid directive, found {self.curr_tok.value} instead; expected valid directive: {DIRECTIVES}")

    def parse_keyed_memory_access(self):
        self.expect(TokenType.IDENTIFIER)
        this = self.advance().value.lower()
        if self.expect(TokenType.SYMBOL).value != ']':
            raise ValueError(f'invalid keyed-memory access syntax; expected \"]\", found {self.peek()} at {self.curr_tok.pos_repr()}')

        self.advance() # [
        self.advance() # next token

        if not (this.lower() in MEMORY_ACCESS_NAMES.keys()):
            raise ValueError(f"invalid memory-access identifier at {self.curr_tok.pos_repr()}: {this}")

        return MEMORY_ACCESS_NAMES[this]

    def parse_keyed_size_mod(self):
        self.expect(TokenType.IDENTIFIER)
        this = self.advance().value.lower()
        if self.expect(TokenType.SYMBOL).value != ')':
            raise ValueError(f'invalid size-modifiier access syntax; expected \")\", found {self.peek()} at {self.curr_tok.pos_repr()}')

        self.advance() # (
        self.advance() # next token

        if not (this.lower() in SIZES_NAMES.keys()):
            raise ValueError(f"invalid size-modifier identifier at {self.curr_tok.pos_repr()}: {this}")

        return SIZES_NAMES[this]

    def evaluate_operand_macros_expansion(self, ):
        # maybe some add symbols and create a variable to store what operation should be done
        # like &, *, /, etc, but that's a big ol' to-do there
        adder = 0
        multiplier = 1
        while(self.curr_tok.type in {TokenType.SYMBOL, TokenType.IDENTIFIER, TokenType.INTEGER_LITERAL}):
            if self.curr_tok.type == TokenType.SYMBOL:
                if self.curr_tok.value == '+':
                    multiplier *= 1
                elif self.curr_tok.value == '-':
                    multiplier *= -1
                elif self.curr_tok.value == ')':
                    return adder
                else:
                    raise ValueError(f"unexpected symbol {self.curr_tok.value} found in expression at {self.curr_tok.pos_repr()}")
                self.advance()
            elif self.curr_tok.type == TokenType.IDENTIFIER:
                value = self.resolve_data_declaration(self.curr_tok.value)
                if isinstance(value, str):
                    raise ValueError(f"unexpected string in data declaration {self.curr_tok.value} at {self.curr_tok.pos_repr()}")
                adder += multiplier * value
                multiplier = 1
                self.advance()
            elif self.curr_tok.type == TokenType.INTEGER_LITERAL:
                value = int(self.curr_tok.value)
                adder += multiplier * value
                multiplier = 1
                self.advance()

        raise ValueError(f"Unexpected token at {self.curr_tok.pos_repr()}: {self.curr_tok.tp()}, {self.curr_tok}")

    def resolve_data_declaration(self, decl_name):
        declarations = self.ofile.cursor.section
        if not declarations:
            raise ValueError("this should be unreachable but anyway")

        declarations = declarations.decl

        if not decl_name in declarations:
            raise ValueError(f"Can't resolve declaration: {decl_name} at section {self.ofile.cursor.section.name} (error at {self.curr_tok.pos_repr()})")

        dd = declarations[decl_name].value

        if isinstance(dd, str):
            raise ValueError(f"Expected integer-literal declaration {decl_name} at section {self.ofile.cursor.section.name}, (error at {self.curr_tok.pos_repr()})")
        return dd

    def operand(self):
        # operand parsing was worse than instruction parsing :)

        if self.curr_tok.type in {TokenType.EOL, TokenType.EOF}:
            # and the fact that this if statement is necesary is hard proof of it
            return None

        mem_access = None
        size       = None
        while(self.curr_tok.type == TokenType.SYMBOL):
            if self.curr_tok.value == '[':
                if mem_access: raise ValueError(f"cannot specify memory access modifier twice (at {self.curr_tok.pos_repr()})")
                mem_access = self.parse_keyed_memory_access()
                continue
            if self.curr_tok.value == '(':
                if size: raise ValueError(f"cannot specify size descriptor twice (at {self.curr_tok.pos_repr()})")
                size = self.parse_keyed_size_mod()
                continue
            if self.curr_tok.value in MEMORY_ACCESS_SYMBOLS.keys():
                if mem_access: raise ValueError(f"cannot specify memory access modifier twice (at {self.currr_tok.pos_repr()})")
                mem_access = MEMORY_ACCESS_SYMBOLS[self.curr_tok.value]
                self.advance()
                continue
            if self.curr_tok.value in SIZE_DESCR_SYMB.keys():
                if size: raise ValueError(f"cannot specify size descriptor twice (at {self.curr_tok.pos_repr()})")
                size = SIZE_DESCR_SYMB[self.curr_tok.value]
                self.advance()
                continue
            if self.curr_tok.value == '%':
                self.advance()  # consume '%'
                self.expect([TokenType.SYMBOL], curr=True)
                if self.curr_tok.value != '(':
                    raise ValueError(f"expected '(' after '%', found {self.curr_tok.value} (at {self.curr_tok.pos_repr()})")
                self.advance()  # consume '('
                value = self.evaluate_operand_macros_expansion()
                self.expect([TokenType.SYMBOL], curr=True)
                if self.curr_tok.value != ')':
                    raise ValueError(f"expected ')' after expression, found {self.curr_tok.value} (at {self.curr_tok.pos_repr()})")
                self.advance()  # consume ')'
                return Operand(mem_access or MemoryAccess.DETECT, TokenType.INTEGER_LITERAL, value, size or -1)
            raise ValueError(f"unexpected symbol {self.curr_tok.value} found at argument parsing at {self.curr_tok.pos_repr()}")

        # this token _should_ be either an identifier or a integer literal
        self.expect([TokenType.IDENTIFIER, TokenType.INTEGER_LITERAL], curr = True)

        value = self.curr_tok.value
        type = self.curr_tok.type
        self.advance()

        return Operand(mem_access or MemoryAccess.DETECT, type, value, size or -1)

    def instruction(self):
        og_pos = self.curr_tok.pos.copy()
        ops  = [self.curr_tok.value]
        oper = []
        self.advance()
        while(self.curr_tok.type == TokenType.IDENTIFIER):
            ops.append(self.curr_tok.value)
            self.advance()

        if self.curr_tok.type in { TokenType.INTEGER_LITERAL, TokenType.STRING_LITERAL}:
            raise ValueError(f"unexpected type of token found at end of instruction argument parsing: {self.curr_tok} at {self.curr_tok.pos_repr()}")

        if self.curr_tok.type in { TokenType.EOL, TokenType.EOF }: # TokenType.EOF, so for some reason the file ends in a weirdly specific operation, but
                                                                   # wathever, just bare with it
            if len(ops) - 1:
                # for something like inc eax
                # inc is the instruction and eax the parameter, this code is the one that should take care of this
                return Instruction(Opcode(ops), [
                    Operand(MemoryAccess.DETECT,TokenType.IDENTIFIER, ops.pop(), -1)
                ])

            return Instruction(Opcode(ops), [])

        if self.curr_tok.value == ',' and (len(ops) - 1):
            # inc eax, ...
            oper.append(Operand(MemoryAccess.DETECT, TokenType.IDENTIFIER, ops.pop(), -1))

        if self.curr_tok.value == ',':
            # inc, eax ... (supported, but weird-ass)
            # (or upper function fallback)
            self.advance()

        op = self.operand()
        while(op):
            oper.append(op)
            if self.curr_tok.type == TokenType.SYMBOL and self.curr_tok.value == ',':
                # comma-separated arguments
                self.advance()
            op = self.operand()

        return Instruction(Opcode(ops), oper)
        # self.ofile.write(Instruction(Opcode(ops), oper))

    def parse(self):
        while not self.eof():
            if self.curr_tok.type == TokenType.INTEGER_LITERAL:
                raise ValueError(f"unexpected integer literal {self.curr_tok.value} at {self.curr_tok.repr_pos()}")
            if self.curr_tok.type == TokenType.DIRECTIVE:
                self.directive()
                continue
            if self.curr_tok.type == TokenType.IDENTIFIER:
                self.ofile.write(self.instruction())
                continue
            if self.curr_tok.type == TokenType.EOL:
                self.advance()
                continue
            raise ValueError(f"unexpected type of toke {self.curr_tok.tp()} at {self.curr_tok.repr_pos()}: {self.curr_tok}")
            self.advance()

        return self.ofile

    def add_to(self, ofile:OutputFile) -> None:
        if self.tok_ptr:
            # womp womp, this is already parsed so too bad
            raise ValueError(f"cannot append already parsed code to the OutputFile")


        # this lowkey smart af (untested)
        self.ofile = ofile # anyway, we're just writing into existing code already,
                           # so as long as our ofile is empty, this should be fine

# """
# from here on now, this shall be very documented, because this is the user-api

# this is the Dumper baseclass implementation, see help(Dumper) to check on how it works

# """


# class Endiannes(IntEnum):

#     """
#     this is used to mark the architecture's endiannes

#     """

#     BIG_ENDIAN    = 0
#     LITTLE_ENDIAN = 1

# class SymbolTypes(IntEnum):
#     # symbol to a function
#     FUNCTION = 0
#     # symbol to a label
#     LABEL = 3

#     # symbol to a variable
#     # example:
#     # .section data:
#     #     my_var: (db) 0x30
#     # .section code:
#     #     ...
#     #     mov eax, #my_var ; here, #my_var is referencing the value at my_var, this should Create a Symbol(SymbolTypes.VARIABLE, ...)
#     VARIABLE = 1
#     # symbol to a constant
#     # example:
#     # .section data:
#     #      .db my_var 0x30 ; this is not declared to be a variable but just told to put here that byte and create a Symbol which ; Wow, this syntax is old
#     #                      ; points to this, this symbol should have type Symbol(SymbolTypes.CONSTANT)
#     CONSTANT = 2

#     # pointer to a variable/constant
#     # example:
#     # .section data:
#     #     .db my_const 0x30
#     #      my_var: (db) 0x30
#     # .section code:
#     #     .func rewrite:
#     #         str *my_var, #0x40    ; *my_var should create a Symbol(SymbolTypes.PT_DATA) because it's a pointer to a data symbol
#     #         str *my_const, #0x44  ; I wouldnt mutate the binary's code at runtime, but this also creates a Symbol(SymbolTypes.PT_DATA) to my_const
#     PT_DATA = 5

#     # example
#     # .section data: ...
#     # .section code:
#     #     ....
#     #     jmp data
#     SECTION = 4

# class Symbol(object):
#     """
#     base class for any code-defined symbol, this shall be inherited if necesary.
#     """
#     type: SymbolTypes
#     name:str
#     section:Section

#     size: Optional[int]

#     def __init__(self, type:SymbolTypes, name:str, section:Section):
#         self.type = type
#         self.name = name
#         self.section = section

#         self.size    = None
#         self.address = None # not calculated

# class Reference(object):
#     """
#     this represents a reference to an address that shall be stored in the OutputFile but not actually dumped
#     the linker shall take care of this ones
#     """
#     def __init__(self, symbol):
#         self.symbol = symbol

#     def __len__(self):
#         if self.symbol.size is None:
#             raise ValueError(f"cannot compute object's size")

#         return self.symbol.size

# class RawCode(object):

#     _elements:List[Union[bytes, Reference]]

#     def __init__(self, *adders):
#         self._elements = []
#         for adder in adders:
#             self += adder

#     def __iadd__(self, other):
#         if isinstance(other, bytes):
#             last = self._elements.pop() if len(self._elements) else None
#             if isinstance(last, bytes):
#                 self._elements.append(last + other)
#                 return self
#             self._elements.append(last) if last else None
#             return self._elements.append(other) or self
#         elif isinstance(other, Reference):
#             return self._elements.append(other) or self
#         elif isinstance(other, RawCode):
#             self._elements.extend(other._elements)
#             return self
#         return NotImplemented

#     def _iter(self):
#         for obj in self._elements:
#             if isinstance(obj, bytes):
#                 yield obj.hex(' ')
#             else:
#                 yield '<' + obj.symbol.name + '>'

#     def __str__(self):
#         return " ".join(self._iter())

#     __repr__ = __str__

# class Dumper(object):
#     """
#     Dumper baseclass, shall be inherited with this is the one that takes care of assembling
#     basically anything, but the architecture can be user-defined


#     this class will take care of almost all the logic but actually encoding the instructions into machine code
#     which shall be dumped into the object file.
#     """

#     # if set to false, automatically raise an error when one found
#     SUPPORT_COMPOSED_INSTRUCTIONS = True

#     def __init_subclass__(cls, *a,**k):
#         callback = {}
#         for o_name, o_val in cls.__dict__.items():
#             if not callable(o_val):
#                 continue
#             if not hasattr(o_val, '__opcode__'):
#                 continue

#             op = o_val.__opcode__
#             if isinstance(op, list):
#                 curr = callback
#                 for v in op:
#                     if not v in curr:
#                         curr[v] = {}
#                     curr = curr[v]
#                 curr[v] = o_val
#             else:
#                 callback[op] = o_val

#         cls._inst_dumpers = callback

#     def __init__(self, ofile:OutputFile) -> None:
#         self.ofile = ofile

#     def dump(self, objfile:Optional[OutputFile] = None) -> OutputFile:
#         """

#         of = OutputFile()
#         ...
#         my_dumper.dump(of)

#         will dump into `of` and return `of`; if no OutputFile is given, the dumper creates it

#         of = my_dumper.dump(of)
#         """
#         if objfile is None:
#             objfile = OutputFile()

#         # sets up the most basic definitions (functions, sections and data declarations) just as symbols
#         self._register_sections(objfile)


#         for section in self.ofile.sections.values():
#             self._section_process(section)

#     def _section_process(self, section:Section) -> None:
#         for f_name, f_code in section.functions.items():
#             self._function_process(f_code)

#     def _search_for_opcode(self, opcode):
#         if opcode.composed and not type(self).SUPPORT_COMPOSED_INSTRUCTIONS:
#             raise ValueError(f"opcode {opcode.name} is not supported because the machine does not support composed instructions")

#         callbacks = getattr(self, '_inst_dumpers', None)
#         if callbacks is None or callbacks == {}:
#             raise ValueError(
#                 f"could not find ANY opcode dumper for the dumper {self.__class__.__name__}, please, check again your code")

#         for op in opcode.opcode:
#             if callable(callback): # last one reached before the end of the composed opcode
#                 raise ValueError(
#                     f"invalid opcode: {opcode.name}, could not load any dumper for it")
#             if not op in callback.keys():
#                 raise ValueError(
#                     f"invalid opcode: {opcode.name}, could not load any dumper for it")
#             callback = callback[op]

#         return callback

#     def _function_process(self, function: Function) -> None:
#         for instruction in function.instr:
#             opcode = instruction.opcode
#             callback = self._search_for_opcode(opcode)
#             instruction.code = callback(instruction)

#     def _raw_symbol_table_update(self, objfile:OutputFile, sym_name:str, symbol:Symbol, overwrite:bool = True) -> None:
#         _sentinel = object()
#         if not overwrite and not (objfile.sym_table.get(sym_name, _sentinel) is _sentinel):
#             raise ValueError(f"cannot overwrite symbol {sym_name} at OutputFile")
#         objfile.sym_table[sym_name] = symbol


#     def _register_sections(self, objfile:OutputFile) -> None:
#         """ register all named sections at OutputFile into OutputFile """
#         if objfile.sections != {}:
#             raise NotImplementedError("merge two object files")
#         objfile.sections = self.ofile.sections # this should do for now

#         # create symbols to all sections (everything going to point here for now on)
#         for secname, section in self.ofile.sections.items():
#             # iter through all sections
#             # no need to overwrite = False bc we're updating from a dict so there cannot be any intersections.
#             self._raw_symbol_table_update(objfile, secname, Symbol(SymbolTypes.SECTION, secname, section))
#             self._register_functions_of(objfile, section)

#     def _register_functions_of(self, objfile:OutputFile, section:Section) -> None:
#         # sets up all the Symbols to the respective functions.
#         for funcname, funct in section.functions.items():
#             self._raw_symbol_table_update(objfile, funcname, Symbol(SymbolTypes.FUNCTION, funcname, section), overwrite = False)

# def instruction_dumper(opcode):

#     def decorator(func):
#         def wrapper(*a,**k):
#             r = func(*a,**k)
#             if not isinstance(r, RawCode):
#                 raise ValueError(f"@instruction_dumper must always return an instance of RawCode()")
#             return r

#         wrapper.__opcode__ = opcode
#         return wrapper

#     return decorator


def tok(t:str) -> OutputFile:
    # just a shortcut
    return Parser(Tokenizer(InputBuffer(t)).tokenize())

# idk why this is not commented, fr, don't use this shit

class Reference(object):
    symbol:Symbol
    size:int
    def __init__(self, symb, size):
        self.symb = symb
        self.size = size

class Code(object):

    _code:List[Union["Code", bytes, Reference]]
    
    def __init__(self):
        self._code = []
        self._base_address = 0
    
    def _write_bytes(self, btes):
        if isinstance(btes, int):
            btes = btes.to_bytes(((btes.bit_length() // 8) + (1 if btes.bit_length() % 8 else 0)), 'little')
        if self._code and isinstance(self._code[-1], bytes):
            self._code[-1] = self._code[-1] + btes
            return
        self._code.append(btes)
    
    def _write_code(self, obj):
        # maybe more logic in the future
        for o in obj._code:
            self += o
    
    def _write_reference(self, obj):
        self._code.append(obj)
    
    def write(self, obj):
        # write an object into the `code`
        if isinstance(obj, (bytes, int)):
            self._write_bytes(obj)
        elif isinstance(obj, Code):
            self._write_code(obj)
        elif isinstance(obj, Reference):
            self._write_ref(obj)
        return self
        
    __iadd__ = write
        
def ofile_print_info(ofile):
    print("Starting OutputFile report")
    for sec_name, section in ofile.sections.items():
        print('    +---- Printing memory section:', sec_name)
        for f_name, f_code in section.functions.items():
            print('        +---- Printing function', f_name,':')
            print('            +---- Function attributes')
            for attrname, attrval in f_code.meta.items():
                print('                +---- .attr', attrname, '=', attrval)
            print('            +---- Function code:')
            for il, instr in enumerate(f_code.instr):
                print(f'                +---- instr[{il}]-opcode:', instr.opcode.name)
                for index, oper in enumerate(instr.operands):
                    print(f'                    +---- instr[{il}]-operand[{index}]:', oper)


# class ExampleDumper(Dumper):
#     @instruction_dumper('nop')
#     def nop_encode(self, nop_instruction:Instruction) -> RawCode:
#          code = RawCode(b'\x00')
#          return code

#     @instruction_dumper('ret')
#     def ret_encode(self, ret_instruction:Instruction) -> RawCode:
#         return RawCode(b'\x09') # for example # damn this old, it's universally \x45

#     @instruction_dumper('jmp')
#     def jmp_encode(self, jmp_instruction:Instruction) -> RawCode:
#         code = RawCode(b'\x1F')
#         if len(jmp_instruction.operands) != 1:
#             raise ValueError(f"invalid jmp usage")
#         op = jmp_instruction.operands[0]


# class LayoutPrinter(object):
#     def __init__(self, ofile):
#         self.ofile = ofile
#         self.section = None
#         self.function = None
#         self.label = None

class AssemblyLayoutPrinter(object):
    def __init__(self, ofile, stdout:io.StringIO):
        # parameters
        self.ofile = ofile
        self.stdout = stdout

        # formatting parameters (prob should make a class out of this)
        self.address = 0
        self.bytecode_area = 12
        self.label_area = 20
        self.opcode_area = 8
        self.operand_area = 50
        self.d_decl_area = 40
        self.s_decl_area = 4
        self.s_decl_value_area = 20
        # at least this was cleaner and used a fucking dict
        self.symb_table_field_lengths = {
            "name":40,
            "type":30,
            "obj":40,
            "obj-type":30,
            "linked":40,
            'audit':10,
            'attr':9,
            'org':20,
            'addr':10,
            'object':20
        }
        self.fields = ["name", "type", "obj-type", "linked", "audit",'org','object']
        # maybe use getattr instead? idk rlly
        self.fields_compute = {
            "name":lambda name, symbol:     name,
            "type":lambda name, symbol:     symbol.type.name,
            "obj-type":lambda name, symbol: type(symbol.obj).__name__ + (f'<{symbol.obj.size}>' if isinstance(symbol.obj, DataDefinition) else ''),
            "audit":lambda name, symbol:    str(symbol.obj.audit) if (hasattr(symbol.obj,'audit') and isinstance(symbol.obj.audit, int)) else '-1',
            "linked":lambda name, symbol:   str(symbol.linked),
            'attr':lambda name, symbol:     self._symb_table_attr_repr(symbol),
            "org":lambda name, symbol:      self._symb_table_org(symbol),
            "addr":lambda name, symbol:     str(symbol.address),
            'object':lambda name, symbol:   symbol.object.name
        }
        self._posible_symb_table_fields = list(self.fields_compute.keys())

    def _symb_table_attr_repr(self, symb):
        if not isinstance(symb.obj, Function):
            return "~"

        fm = symb.obj.meta
        # for now, no other posible attributes
        return ("E" if "glob" in fm else '-') + '------'

    def _symb_table_org(self, symb):
        if not isinstance(symb.obj, (Function, Section)):

            # can't possibly have an `org`
            return "~"
        # org set to -1 by default
        return hex(symb.obj.org)

    def process_function(self, func):
        # display instructions
        self._assembly_display_center(f'{"FUNCTION" if not func.meta.get("external", False) else "FUNCTION(external)"} {func.name}')
        instructions = func.instr
        il_t = {v:k for k,v in func.labels.items()}

        bytecode_area = self.bytecode_area
        label_area = self.label_area
        opcode_area = self.opcode_area
        operand_area = self.operand_area

        for instr_ptr, instr in enumerate(instructions):
            bytecode = instr.code
            size = 1
            if bytecode:
                # code already compiled
                size = len(bytecode)
                self.address += size
                bytecode = bytecode.hex(' ')
            else:
                bytecode = ' ' * bytecode_area

            label = il_t.get(instr_ptr, -1)
            if label == -1:
                label = label_area * ' '
            bytecode = self.clamp(bytecode, bytecode_area)
            label = self.clamp(label, label_area)
            opcode = self.clamp(instr.opcode.name, opcode_area)

            operand_repr = " , ".join(
                    map(self._operand_repr, instr.operands)
                )
            operand_repr = self.clamp(operand_repr, operand_area)
            if instr.code:
                # make up for pre-added size
                self.emit(f"{(self.address - size):04X} {bytecode} {label} {opcode} {operand_repr}")
                continue

            self.emit(f"{self.address:04X} {bytecode} {label} {opcode} {operand_repr}")
            self.address += 1

    def _operand_repr(self, operand):
        value = operand.value
        if isinstance(value, int):
            value = hex(value)

        s = {v:k for k,v in SIZES_NAMES.items()}.get(operand.size, 'dU')
        mt = {v:k for k,v in MEMORY_ACCESS_SYMBOLS.items()}.get(operand.memory_access, '/')
        t =  ' ' + s + ' ' + mt + value
        return t

    def clamp(self, message, length, add = 1):
        if len(message) > length:
            return message[:(length - 2)] + '..'
        return message + ((length - len(message)) * ' ' if add else '')

    def process_data_declaration(self, decl):
        name = decl.name
        size = {v:k for k,v in SIZES_NAMES.items()}.get(decl.size, 'dU')
        value = decl.value

        if isinstance(value, int):
            value = hex(value)

        self.emit(f'{self.address:04X} {self.clamp(name, self.d_decl_area)} {size} {self.clamp(value, self.s_decl_value_area)}')
        self.address += decl.size


    def process_other(self, other):
        if not isinstance(other, Auditable):
            return # how tf this ended here

        # only DataDefinition and Function inherit from Auditable
        self.process_data_declaration(other)

    def emit(self, msg:str):
        self.stdout.write(msg + '\n')

    def _assembly_display_center(self, msg:str):
        self.stdout.write(msg.center(self.bytecode_area + self.label_area + 4 + self.operand_area + self.opcode_area + 4) + '\n')

    def process_section(self, section:Section):
        # display section
        self._assembly_display_center(f'SECTION {section.name}')
        for obj in section.audit:
            if isinstance(obj, Function):
                # display function
                self.process_function(obj)
                continue
            # display whatever other Auditable
            self.process_other(obj)

    def printout(self, header = False):
        # todo: print-header if posible (for address as it occupies 4 chars)

        for section in self.ofile.sections.values():
            self.process_section(section)

    def symbol_table_printout(self, field_lengths = None, fields = None,sep=" | ", center=True):
        # this is way cleaner than printout() but wathever // nana, ruined it with custom fields to the table

        center = not center # argument passed to add so it has to be negated

        # some argument pre-processing
        field_lengths = field_lengths or self.symb_table_field_lengths
        fields = fields or self.fields
        if isinstance(fields, str) and fields.lower() == 'all':
            fields = self._posible_symb_table_fields
        elif isinstance(fields, str):
            raise ValueError(":(")

        # compute the header in a single line real quick
        header = sep.join(map(
                lambda field: self.clamp(string.capwords(field),
                    field_lengths.get(field, self.symb_table_field_lengths[field]),
                    add = 0
                ).center(field_lengths.get(field, self.symb_table_field_lengths[field])),
                fields
            ) # map
        ) # join

        # emit header
        self.emit(header)
        self.emit("=" * len(header))

        # iter symbols
        for name, symbol in self.ofile.symb_table.items():
            # fetch field representation functions
            fmap = map(lambda field: self.fields_compute.get(field,lambda n,s:"<unknown-field>"),
                        fields
                    )
            # map'em
            u = list([f(name, symbol) for f in fmap])
            # join and format everything
            line = sep.join(
                    map(
                        # first lambda to unpack
                        lambda a:(
                            lambda i,f: (
                            # index-accessing field at the 'u' array
                            # u[i] is the actuall field data given
                            # i is the index
                            # f is the field name
                            # this screams "size = field_lengths.get(f, self.symb_table_field_lengths[f]" but
                            # whatever
                            self.clamp(u[i],
                                # make sure it's a suitable length
                                field_lengths.get(f, self.symb_table_field_lengths[f]),
                                # center is negated, with center, don't add padding, without center add padding and the .center() won't do anything
                                add = center
                            # the call for .center()
                            ).center(field_lengths.get(f, self.symb_table_field_lengths[f]))
                        # arg-mapping stuff
                        # unpacks the arguments
                        ))(*a),
                        # map through index a and field of fields
                        enumerate(fields)
                    )
                )
            # emit line
            self.emit(line)

