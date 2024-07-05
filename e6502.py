import sys
import io
import os
import math
# from blogging import record

# address fetching so slow gotta use this
# from functools import lru_cache

# maybe?
# from numpy import array, uint8
# if I'm having some serious memory issues then sure

def twos_complement(value, bit_width):
    # Ensure value is within the bit width range
    return ((1 << bit_width) - value) & ((1 << bit_width) - 1)

def decargspec(argspec: int):
    # if dual:
        # a =  1 << ((argspec >> 6) + 1), (argspec >> 4) & 3
    # argspec = argspec & 0xFF    
    # asp = argspec >> 4
    # if dual:
        # return 1 << ((asp >> 2) + 1), (asp) & 3, 1 << ((bsp >> 2) + 1), (bsp) & 3
    return 1 << ((argspec >> 6)), (argspec >> 4) & 3, argspec & ((1 << 4) - 1)

def dualdecargspec(argspec):
    return 1 << ((argspec >> 6)), (argspec >> 4) & 3, 1 << ((argspec & 12) >> 2), (argspec & 3)

def sign_extend(v, sz):
    raise ValueError("sign extending not yet implemented :p")    

class MemoryRange(object):
    def __init__(self,start, end, size,name = None):
        self.start = start
        self.end = end
        self.size = size
        self.name = name or '<unnamed>'

        if(self.end - self.start != self.size):
            raise ValueError("end - start != size for memory space " + self.name)
        if(self.size < 1):
            raise ValueError("size of section " + self.name + " smaller than 1 byte long")

    @classmethod
    def from_size(cls, size, start = 0, name = None):
        return MemoryRange(start, start + size, size, name)

    @classmethod
    def from_end(cls, start, end, name=None):
        return MemoryRange(start, end, end - start, name)

def binary_search(arr, x):
    i = 0
    while(len(arr) > i and arr[i]<x):
        i += 1
    if(len(arr) <= i):
        return i - 1
    return i


class LazyMemory(object):
    def __init__(self, mem_range):
        self.mem_range = mem_range
        self._computed_memories = {}
        self._computed_memories_starts = set()
        self._computed_memories_starts_l = list()

    MEMSEP_THRESHOLD = 256
    MEMCHUNK_ALLOC = 64

    MERGE = False

    @property
    def size(self):
        return self.mem_range.size

    @property
    def end(self):
        return self.mem_range.end

    @property
    def start(self):
        return self.mem_range.start


    # @record("M")
    def _memory_bin_search(self, address, _range = None):
        # def _memory_bin_search(self, address):
        # print(self._computed_memories_starts_l)

        return binary_search(_range or self._computed_memories_starts_l, address)

    # @lru_cache(maxsize=128)
    def _ready_address_chunk(self, address, size, alloc = True):
        
        # looking in retrospective, this is straight up one of my best-made functions
        # even thought it took some serious effort, (you can guess that up for the amount of commented if statements)
        # but it works really well for caching memory that's very far away
        # _and_ it shouldnt be that slow, ¿should it?
        # I'll probably make a stress test on the future
        
        # address from 0 to mem_range.size
        # print(address, size)
        size = min(size, (self.size - address))
        # there's no memory loaded yet
        if not len(self._computed_memories_starts_l):
            # print("X")
            if not alloc:
                return None
            # print(address)
            self._computed_memories_starts.add(address)
            self._computed_memories_starts_l.append(address)
            self._computed_memories[address] = [0] * size
            return address
        # print(self._computed_memories_starts_l)
        _nearest  = self._memory_bin_search(address) # cual es el indice  a direccion de memoria mas cercana, pero menor que address
        # if _nearest > len(self._computed_memories_starts_l) - 1:
            # _nearest -= 1
        # i(_nearest, self._computed_memories_starts_l)
        # print(_nearest)
        addr = self._computed_memories_starts_l[_nearest] # cogemos la direccion de memoria mas cercana pero menor (Addr)

        while(addr > address):
            _nearest -= 1
            addr = self._computed_memories_starts_l[_nearest] # cogemos la direccion de memoria mas cercana pero menor (Addr)

        # print(addr)
        addr_size = len(self._computed_memories[addr]) # cuanto hay alocado en `addr`
        # print(addr + addr_size, address + size)
        # print(addr, addr_size, address, size)
        # print("address", address, "gonna map to address", addr,"chunk of", addr_size, 'bytes alloc', alloc)
        # print(addr, addr_size, address, size)
        # if(addr > address) and not alloc:
            # return None

        if (addr + addr_size >= address + size) and not(addr > address): # ya deberia estar allocado
            # already allocated space, Yai
            
            return addr
        # hay que allocarlo
        # we know we gotta allocate it if this is executed
        if not alloc:
            return None

        required_space = address + size - ( addr + addr_size)

        _next_addr = -1
        _next = -1
        dist = math.inf

        if _nearest != len(self._computed_memories_starts_l) - 1:
            # if there's next, fetch next
            # print("X")
            _next = _nearest + 1
            _next_addr = self._computed_memories_starts_l[_next]
            # calculate distance between this address and the next one
            dist = _next_addr - address
            # print(_next_addr, addr)
            # size = min(size, (_next_addr - addr))
            # required_space = dist
            # print(dist)

        #              top required addr           top allocated address
        # print(address, size, addr, addr_size)
        # print("required_space:",required_space)
        # print(required_space)
        # crear un nuevo elemento diccionario, un nuevo chunk de memoria


        if(required_space >= type(self).MEMSEP_THRESHOLD) or required_space < 0:
            if dist < size:
                if self.MERGE:
                    # merge this one and next
                    # this shouldnt be done a lot of times, maybe I'll add up some more logic to 
                    # instead allocate up the addr allocate up to addr - MEMCHUNK_ALLOC to make 
                    # sure everytime you go down in memory it doesnt do this crazy computations all
                    # again over and over
                    self._computed_memories_starts.remove(_next_addr)
                    self._computed_memories_starts_l.remove(_next_addr)

                    space = self._computed_memories[_next_addr]

                    new = [0] * dist
                    new.extend(space)
                    del(self._computed_memories[_next_addr])

                    size = -1
                else:
                    # this should be crazy fast though
                    size = dist
            self._computed_memories_starts.add(address) # still I don't know why I keep the set alive (it's pretty much not used)
            self._computed_memories_starts_l.insert(_nearest + 1, address)
            if size > 0:
                self._computed_memories[address] = [0] * size
            else:
                # this can only be the case on the merged code case
                self._computed_memories[address] = new
            return address

        # o ampliar el ya hecho
        addr_space = self._computed_memories[addr]
        while(required_space):
            addr_space.append(0x00)
            required_space -= 1

        return addr

        # if addr + addr_size < address:
            # while(addr + addr_size - size >= address):
                # self._computed_memories[address].append(0x00)
                # addr_size += 1
            #
            # return addr

    def _fetch_memory(self, address):
        # print(address)
        if address > self.size:
            return 0x00
        addr = self._ready_address_chunk(address, 1, alloc = False)
        # print(addr)
        if addr is None:
            # if the address hasnt been allocated then it's `default byte` (0)
            return 0x00
            # jaja you idiot
            # addr = self._ready_address_chunk(address, type(self).MEMCHUNK_ALLOC, alloc = True)
        a = self._computed_memories[addr]
        # print(a)
        return self._computed_memories[addr][address - addr]

    def _set_memory(self, address, val):
        if address >= self.size:
            return
        if(not isinstance(val, int) or val & 0xFF !=  val):
            raise ValueError("val not a byte and blah blah blah")
        addr = self._ready_address_chunk(address, 1, alloc = False)
        # print(addr)
        if addr is None:
            addr = self._ready_address_chunk(address, self.MEMCHUNK_ALLOC, alloc = True)
        # print(self._computed_memories[addr],address, addr, address - addr)
        self._computed_memories[addr][address - addr] = val

    def fetch(self, address):
        if address < self.start:
            raise ValueError("address < start")
        if address > self.end:
            raise ValueError("address > end")
        # print(address)
        return self._fetch_memory(address - self.start)

    def set(self, address, val):
        if address < self.start:
            raise ValueError("address < start")
        if address > self.end:
            raise ValueError("address > end")

        return self._set_memory(address - self.start, val)

    def __getitem__(self, address):
        return self.fetch(address)

    def __setitem__(self, address, val):
        self.set(address, val)

    def __repr__(self):
        return f"memory(start={self.start}, end={self.end}, size={hex(self.size)})"


class DirectMemory(LazyMemory):


    # this probably gonna go through so much refactoring
    # like this shit is real bad, at the very design



    PAGE_SIZE = 0xffff

    def __init__(self, *a, **k):
        LazyMemory.__init__(self, *a, **k)
        self.curr = None
        self.curr_addr = None
        # assert not(self.size % type(self).PAGE_SIZE), "nerd't you"

    def reset(self, vector):
        # set the current 0xFFFF page to wherever this is
        if self.curr:
            del self._computed_memories[self.curr_addr]
            self._computed_memories_starts.remove(self.curr_addr)
            self._computed_memories_starts_l.remove(self.curr_addr)
        if vector + 0xFFFF > self.end:
            raise ValueError(f"can't build page over vector: {hex(vector)}")
        if vector < self.start:
            raise ValueError(f"can't build page over vector: {hex(vector)}")
        # this can't be easily modified so that it can support
        # for non-loaded pages to be kept alive
        self.curr_addr = vector
        self._ready_address_chunk(vector, type(self).PAGE_SIZE, alloc = True)
        self.curr = self._computed_memories[self.curr_addr]

    def _fetch_memory(self, address):
        # print(address)
        if address > type(self).PAGE_SIZE:
            raise ValueError
        if not self.curr:
            return 0x00
        # addr = self._ready_address_chunk(address, 1, alloc = False)
        # # print(addr)
        # if not addr:
        #     # jaja smart nerdy stuff
        #     return 0x00
        #     # addr = self._ready_address_chunk(address, type(self).MEMCHUNK_ALLOC, alloc = True)
        # a = self._computed_memories[addr]
        # # print(a)
        # return self._computed_memories[addr][address - addr]
        return self.curr[address]

    def _set_memory(self, address, val):
        if address > type(self).PAGE_SIZE:
            raise ValueError
        if(not isinstance(val, int) or val & 0xFF !=  val):
            raise ValueError("val not a byte and blah blah blah")
        if not self.curr:
            return
        # addr = self._ready_address_chunk(address, 1, alloc = False)
        # # print(addr)
        # if not addr:
        #     addr = self._ready_address_chunk(address, type(self).MEMCHUNK_ALLOC, alloc = True)
        # # print(self._computed_memories[addr],address, addr, address - addr)
        # self._computed_memories[addr][address - addr] = val
        self.curr[address] = val

    fetch = _fetch_memory

    set = _set_memory

class RegisterPage:
    """
    
    A =       RAL           +          RAH           = 
        PAL    +    PAH     +    QAL    +    QAH     = 
     WAL + WAH + XAL + XAH  + YAL + YAH + ZAL + ZAH


    """
    def __init__(self):
        self._64bit_registers = {
            0: 0,  # a
            1: 0,  # b
            2: 0,  # c
            3: 0   # d
        }
        self._32bit_registers = {
            0: 0,  # R0
            1: 0,  # R1
            2: 0,  # R2
            3: 0   # R3
        }

    def get_register(self, address):
        first_bit = (address >> 7) & 1
        if first_bit:
            index = address & 3
            return self._32bit_registers[index]
        index = (address >> 5) & 3
        sub_index = address & 0x1F
        return self._get_64bit_subregister(index, sub_index)

    def set_register(self, address, value):
        first_bit = (address >> 7) & 1
        if first_bit:
            index = address & 3
            self._32bit_registers[index] = value & 0xFFFFFFFF  # Ensure it's 32-bit
            return
        index = (address >> 5) & 3
        sub_index = address & 0x1F
        self._set_64bit_subregister(index, sub_index, value)

    def _get_64bit_subregister(self, index, sub_index):
        reg = self._64bit_registers[index]
        sub_index = sub_index & 0b1111 # clear fifth bit as it really isnt used
        if sub_index == 0:
            return reg
        elif sub_index == 1:
            return reg & 0xFFFFFFFF
        elif sub_index == 2:
            return (reg >> 32) & 0xFFFFFFFF
        elif sub_index == 3:
            return reg & 0xFFFF
        elif sub_index == 4:
            return (reg >> 16) & 0xFFFF
        elif sub_index == 5:
            return (reg >> 32) & 0xFFFF
        elif sub_index == 6:
            return (reg >> 48) & 0xFFFF
        elif sub_index == 7:
            return reg & 0xFF
        elif sub_index == 8:
            return (reg >> 8) & 0xFF
        elif sub_index == 9:
            return (reg >> 16) & 0xFF
        elif sub_index == 10:
            return (reg >> 24) & 0xFF
        elif sub_index == 11:
            return (reg >> 32) & 0xFF
        elif sub_index == 12:
            return (reg >> 40) & 0xFF
        elif sub_index == 13:
            return (reg >> 48) & 0xFF
        elif sub_index == 14:
            return (reg >> 56) & 0xFF

    def _set_64bit_subregister(self, index, sub_index, value):
        reg = self._64bit_registers[index]
        sub_index = sub_index & 0b1111 # clear fifth bit as it really isnt used
        if sub_index == 0:
            self._64bit_registers[index] = value & 0xFFFFFFFFFFFFFFFF
        elif sub_index == 1:
            self._64bit_registers[index] = (reg & 0xFFFFFFFF00000000) | (value & 0xFFFFFFFF)
        elif sub_index == 2:
            self._64bit_registers[index] = (reg & 0x00000000FFFFFFFF) | ((value & 0xFFFFFFFF) << 32)
        elif sub_index == 3:
            self._64bit_registers[index] = (reg & 0xFFFFFFFFFFFF0000) | (value & 0xFFFF)
        elif sub_index == 4:
            self._64bit_registers[index] = (reg & 0xFFFFFFFF0000FFFF) | ((value & 0xFFFF) << 16)
        elif sub_index == 5:
            self._64bit_registers[index] = (reg & 0xFFFF0000FFFFFFFF) | ((value & 0xFFFF) << 32)
        elif sub_index == 6:
            self._64bit_registers[index] = (reg & 0x0000FFFFFFFFFFFF) | ((value & 0xFFFF) << 48)
        elif sub_index == 7:
            self._64bit_registers[index] = (reg & 0xFFFFFFFFFFFFFF00) | (value & 0xFF)
        elif sub_index == 8:
            self._64bit_registers[index] = (reg & 0xFFFFFFFFFFFF00FF) | ((value & 0xFF) << 8)
        elif sub_index == 9:
            self._64bit_registers[index] = (reg & 0xFFFFFFFFFF00FFFF) | ((value & 0xFF) << 16)
        elif sub_index == 10:
            self._64bit_registers[index] = (reg & 0xFFFFFFFF00FFFFFF) | ((value & 0xFF) << 24)
        elif sub_index == 11:
            self._64bit_registers[index] = (reg & 0xFFFFFF00FFFFFFFF) | ((value & 0xFF) << 32)
        elif sub_index == 12:
            self._64bit_registers[index] = (reg & 0xFFFF00FFFFFFFFFF) | ((value & 0xFF) << 40)
        elif sub_index == 13:
            self._64bit_registers[index] = (reg & 0xFF00FFFFFFFFFFFF) | ((value & 0xFF) << 48)
        elif sub_index == 14:
            self._64bit_registers[index] = (reg & 0x00FFFFFFFFFFFFFF) | ((value & 0xFF) << 56)

    def register_from_prefix(self, reg:str):
        reg = reg.lower()

        if reg in {'r0', 'r1', 'r2', 'r3'}:
            return 0b10000000 + ['r0', 'r1', 'r2', 'r3'].index(reg)

        if reg in {'a','b','c','d'}:
            return ['a','b','c','d'].index(reg) << 5

        return (self.register_from_prefix(reg[1]) | {
            'r':{'h':0x2,'l':0x1},
            'p':{'h':0x4,'l':0x3},
            'q':{'h':0x6,'l':0x5},
            'w':{'h':0x8,'l':0x7},
            'x':{'h':0xA,'l':0x9},
            'y':{'h':0xC,'l':0xB},
            'z':{'h':0xE,'l':0xD}
        }[reg[0]][reg[2]])

    def register_size(self, rn):
        if (rn >> 7) & 1:
            # 4 bytes R0, R1, R2 or R3, we don't care
            return 4

        i = rn & ((1 << 5) - 1)
        if i >= 7:
            return 1
        if i >= 3:
            return 2
        if i >= 1:
            return 4
        return 8

    @property
    def a(self):
        return self._64bit_registers[0]

    @a.setter
    def a(self, value):
        self._64bit_registers[0] = value & 0xFFFFFFFFFFFFFFFF  # Ensure it's 64-bit

    @property
    def b(self):
        return self._64bit_registers[1]

    @b.setter
    def b(self, value):
        self._64bit_registers[1] = value & 0xFFFFFFFFFFFFFFFF  # Ensure it's 64-bit

    @property
    def c(self):
        return self._64bit_registers[2]

    @c.setter
    def c(self, value):
        self._64bit_registers[2] = value & 0xFFFFFFFFFFFFFFFF  # Ensure it's 64-bit

    @property
    def d(self):
        return self._64bit_registers[3]

    @d.setter
    def d(self, value):
        self._64bit_registers[3] = value & 0xFFFFFFFFFFFFFFFF  # Ensure it's 64-bit

    @property
    def R0(self):
        return self._32bit_registers[0]

    @R0.setter
    def R0(self, value):
        self._32bit_registers[0] = value & 0xFFFFFFFF  # Ensure it's 32-bit

    @property
    def R1(self):
        return self._32bit_registers[1]

    @R1.setter
    def R1(self, value):
        self._32bit_registers[1] = value & 0xFFFFFFFF  # Ensure it's 32-bit

    @property
    def R2(self):
        return self._32bit_registers[2]

    @R2.setter
    def R2(self, value):
        self._32bit_registers[2] = value & 0xFFFFFFFF  # Ensure it's 32-bit

    @property
    def R3(self):
        return self._32bit_registers[3]

    @R3.setter
    def R3(self, value):
        self._32bit_registers[3] = value & 0xFFFFFFFF  # Ensure it's 32-bit


"""
some random CPU emulator xd

CPU arquitecture made up in real time XD

may do an assembler for it, a very simple one though
"""

Memory = LazyMemory # TODO: non lazy memory xd




def declare_instr(size,addr = 0x01):
    def deco(func):
        func._instr_size = size
        func.opcode = addr
        return func
    
    return deco

class f8403:
    """
    memory ranges:
        0x00000000 <-> 0x01000000 RAM (64Kb)
        0x01000000 <-> 0x08000000 ROM (~)
        0x10000000 <-> 0x11000000 VRAM (64 Kb)
        0x11000000 <-> 0x12000000 I/O ports


        Dp and stack are internal to the CPU so they start at 0x0000
        Dp has to be reseted before usage, and the stack should be initialized, even though I'll probably
        fill the stack pointer with pure trash originally just to "simulate more real cpus"
    """

    HALTING_BIT = (1 << 62)

    OVERFLOW_BIT = (1 << 2)
    ZERO_FLAG    = (1 << 1)

    ALU_CLEAR    = OVERFLOW_BIT | ZERO_FLAG

    opcodes = {}

    POINTER_SIZE = 4 # a pointer is 4 bytes long

    def __init__(self):
        self.ram = Memory(MemoryRange.from_size(0x01000000, start=0, name="ram"))
        self.rom = Memory(MemoryRange.from_end(0x01000000, 0x08000000, name = "rom"))
        self.vram  = Memory(MemoryRange.from_size(0x01000000, start = 0x10000000, name="vram"))



        # this gonna be fun btw
        self.dp = DirectMemory(MemoryRange.from_size(0x000F0000, start=0, name='direct-page'))
        self.stack = Memory(MemoryRange.from_size(0x0000FFFF, name="stack"))

        # another special redirect stuff for I/O ports addressing and stuff
        # just stfu with that for now


        # access it's RAM thing
        self.pc = 0

        self.sr = 0 # status register
        self.sm = 0 # Stack Mode
        self._sp = 0 # stack pointer
        self.instr = 0 # Instruction

        self.cycle = 0


        self.rp = RegisterPage()

        # self.opcodes = {
            # 0x00 : self.nop,
            # 0x40 : self.halt,
            # 0x9A : self.add_rr,
            # 0x9B : self.sub_rr,
            # 0x90 : self.neg_r, 
            # 0x80 : self.direct_page_man,
            # 0x2B : self.push,
            # 0x2A : self.pop, 
            # 0x1A : self.limn,
        # }

        self.interrupt_table = {} # yeh
        self.exec = self._exec
        
        self._opcode_init()
        self.reset()
        
    
    def _opcode_init(self):
        cls = type(self)
        self.opcodes={}
        # sorry for this ugly af line, this only iterates through all the methods of this class
        for method in [(lambda v:(v if callable(v) else None))(getattr(cls, attr)) for attr in dir(cls)]:
            if method == None:
                continue
            if not hasattr(method, 'opcode'):
                continue
            
            self.opcodes[method.opcode] = method

    @property
    def sp(self):
        return self._sp
    
    @sp.setter
    def sp(self, new_val):
        if new_val < 0:
            # print(new_val)
            new_val = self.stack.end + new_val
            # print(new_val, self.stack.end)
            if new_val < 0: raise ValueError("sp < 0")
        self._sp = new_val 


    def read_vector(self, size):
        v  = self.read_address(self.pc, size=size)
        self.pc += size
        # print(size,v)
        return v


    def init_direct_page(self):
        b0 = self.fetch_instruction()
        b1 = self.fetch_instruction()
        b2 = self.fetch_instruction()
        b3 = self.fetch_instruction()

        vector = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)

        self.dp.reset(vector)

        self.cycle += 35

    @declare_instr(1, 0x1A)
    def limn(self):
        self.cycle += 1
        rr = self.fetch_instruction()

        rr_s = self.rp.register_size(rr)

        vector = self.read_vector(rr_s)
        
        self.rp.set_register(rr, vector)
        
        
    @declare_instr(1, 0x90)
    def neg_r(self):
        self.cycle += 1
        wr = self.fetch_instruction()
        
        wr_s = self.rp.register_size(wr)
        self.rp.set_register(
            wr, 
            twos_complement(self.rp.get_register(wr), wr_s * 8)
        )

    @declare_instr("*", 0x80)
    def direct_page_man(self):
        ins = self.fetch_instruction()

        if (ins == 0x05):
            return self.init_direct_page()

        self.cycle += 1
        return

    def address_read(self, addr, inc = 0, size = 1):
        self.cycle += inc

        if addr < 0x01000000:
            # ram
            b =  self.ram
        elif addr < 0x08000000:
            # rom
            self.cycle += 1
            b =  self.rom
        else:
            self.cycle += 2
            b =  self.vram
        
        # as long as `size` is not 4 quadrillion this should be fast enough
        vector = 0
        # print(b, addr)
        while(size):
            vector = (vector << 8) + b[addr + size - 1]
            # print(b, addr, hex(vector), b[addr])
            # addr += 1
            size -= 1
        # print(addr, vector)
        return vector
    
    read_address = address_read
    
    def dp_read(self, addr, size = 0):
        b = self.dp
        vector = 0
        
        while(size):
            vector = (vector << 8) + b[addr + size - 1]
            # addr += 1
            size -= 1
        
        return vector
    
    @declare_instr(2, 0x9B)
    def sub_rr(self, ):
        self.sr = self.sr & ~type(self).ALU_CLEAR # clear the ALU status register's bits

        wr = self.fetch_instruction()
        rr = self.fetch_instruction()

        wr_s = self.rp.register_size(wr)
        rr_s = self.rp.register_size(rr)

        wv = self.rp.get_register(wr)
        rv = self.rp.get_register(rr)
        
        rv = twos_complement(rv, 8 * rr_s) # this function is magnificent

        r = wv + rv
        
        if r >> (wr_s * 8):
            self.sr |= self.OVERFLOW_BIT
        elif r == 0:
            self.sr |= self.ZERO_FLAG
        
        self.cycle += 1

        self.rp.set_register(wr, r)
                

    @declare_instr(2, 0x9A)
    def add_rr(self, ):
        self.sr = self.sr & ~type(self).ALU_CLEAR # clear the ALU status register's bits
        wr = self.fetch_instruction()
        rr = self.fetch_instruction()

        wr_s = self.rp.register_size(wr)
        # rr_s = self.rp.register_size(rr)

        wv = self.rp.get_register(wr)
        rv = self.rp.get_register(rr)

        self.cycle += 1
        r = wv + rv
        # moved to the ALU clear
        # self.sr &= ~(self.OVERFLOW_BIT | self.ZERO_FLAG)
        if r >> (wr_s * 8):
            self.sr |= self.OVERFLOW_BIT
        elif r == 0:
            self.sr |= self.ZERO_FLAG

        self.rp.set_register(wr, r)

    def address_write(self, addr, bt, inc = 0,size = 1):
        self.cycle += inc

        if addr < 0x01000000:
            # ram
            b = self.ram
            # self.ram[addr] = bt
            # return
        if addr < 0x08000000:
            # rom
            self.cycle += 1
            # self.rom[addr] = bt
            b = self.rom
            # return
        else:
            self.cycle += 1
            b = self.vram
        
        while(size):
            b[addr] = (bt & 0xFF)
            bt = bt >> 8
            size -= 1
            addr += 1
        
        return 

    def fetch_instruction(self, inc = 0) -> int:
        # print("fetching", self.pc, ":", self.ram[self.pc])
        self.instr = self.address_read(self.pc, inc = 0)
        self.pc += 1
        return self.instr

    def reset(self):
        # nothing for now
        # put the stack pointer at the end of the stack 
        self.stack.MEMCHUNK_ALLOC = 16 # reduce this
        # self.stack.MERGE = True
        self.stack[0] = 0 # this is to set up variables in stack
        self.sp = self.stack.end
        
    def stack_write(self, val):
        val =  val & 0xFF
        # translation because reading memory backwards is way slower than reading and setting memory forwards
        self.stack[self.stack.end - self.sp] = val
    
    write_stack = stack_write
    
    def stack_read(self):
        return self.stack[self.stack.end - self.sp]
    
    read_stack = stack_read
    
    @declare_instr(2, 0x76)
    def mov(self):
        # mov
        self.cycle += 1
        mode = self.fetch_instruction()
        size,type, rest = decargspec(mode)
        w_reg = self.fetch_instruction()
        w_reg_size = self.rp.register_size(w_reg)
        
        ext = lambda v, sz:v  # zero extend
        if rest:
            ext = sign_extend
            self.cycle += 1
        
        if not (size - 1):
            # size == 1
            if type == 0:
                # register is POINTER
                copy_reg = self.fetch_instruction()
                copy_reg_val = self.rp.get_register(copy_reg)
                
                value_at_address = ext(self.address_read(copy_reg_val, size=w_reg_size), w_reg_size)
                
                self.rp.set_register(w_reg, value_at_address)
                return
            if type == 1:
                # register is DP
                if not self.dp.curr:
                    self.sr |= self.ZERO_FLAG
                    return # dp not initialized
                copy_reg = self.fetch_instruction()
                copy_reg_val = self.rp.get_register(copy_reg)
                
                value = ext(self.dp_read(copy_reg_val, size = w_reg_size), w_reg_size)
                
                self.rp.set_register(w_reg, value)
                return
            if type == 2:
                # inmediate is pointer
                pointer_val = self.read_vector(self.POINTER_SIZE)
                value_at_address = ext(self.address_read(pointer_val, size = w_reg_size),w_reg_size)
                
                self.rp.set_register(w_reg, value_at_address)
                return
            if type == 3:
                dpp_val = self.read_vector(2) # DP address size is 2 bytes
                copy_val = ext(self.dp_read(dpp_val, size = w_reg_size), w_reg_size)
                
                self.rp.set_register(w_reg, copy_val)
                return
            return # this is unreachable                
        if not (type >> 1):
            # print("this is executed")
            copy_reg = self.fetch_instruction()
            # copy the value of copy_reg into w_reg
            self.rp.set_register(w_reg, 
                                 ext(self.rp.get_register(copy_reg), w_reg_size))
            
            return
        # mov inmediate into inmediate
        # print(self.pc)
        older=self.pc
        val  = self.read_vector(w_reg_size)
        # print(self.pc, older+w_reg_size)
        self.rp.set_register(w_reg, ext(val, w_reg_size))
        return
   
    @declare_instr(1, 0x2B)
    def push(self):
        pushed_register = self.fetch_instruction()
        pushed_register_size = self.rp.register_size(pushed_register)

        val_to_push = list(self.rp.get_register(pushed_register).to_bytes(pushed_register_size, 'big')) # inserted on inverted order so this
                                                                                                        # turns to little endian       
        while(pushed_register_size):
            # self.stack[self.sp] = val_to_push[pushed_register_size - 1]
            self.stack_write(val_to_push[pushed_register_size - 1])
            self.sp = self.sp - 1
            pushed_register_size -= 1
            self.cycle += 1

    @declare_instr(1, 0x2a)
    def pop(self):
        write_register = self.fetch_instruction()
        write_register_size = self.rp.register_size(write_register)
        reg = 0
        while(write_register_size):
            # self.stack[self.sp] = val_to_push[pushed_register_size - 1]
            self.sp = self.sp + 1
            # print("before", reg, 'stck', self.read_stack())
            reg = (reg << 8) + self.read_stack()
            # print("after", reg, 'stck', self.read_stack())
            write_register_size -= 1
            self.cycle += 1
        # print(reg)
        self.rp.set_register(write_register, reg)

    @declare_instr(0,0x00)
    def nop(self):
        pass
        # print("nop executed!")

    @declare_instr(0, 0x40)    
    def halt(self):
        self.sr |= self.HALTING_BIT
        
    def opcode_table(self):
        # some chatgpt code to print the table out
        opcode_names = {op: func.__name__.split('_')[0] for op, func in self.opcodes.items()}
        
        # Determine the max length of names for pretty printing
        max_name_length = max(len(name) for name in opcode_names.values())

        # Create the table header
        # print(list([f"0x{i:X}".center(max_name_length) for i in range(0x10)]))
        header = "       " + "   ".join([f"0x{i:X}".center(max_name_length) for i in range(0x10)])
        print(header)
        print("-" * len(header))

        # Create the table rows
        for row in range(0x00, 0xF0 + 1, 0x10):
            row_str = f"{row:X}".center(4)
            for col in range(0x10):
                opcode = row + col
                row_str += ' | '
                if opcode in opcode_names:
                    row_str += opcode_names[opcode].center(max_name_length)
                else:
                    row_str += " " * max_name_length
                # row_str += ' | '
                # row_str += ""
            print(row_str)
            # print('_' * len(row_str))

    def _exec(self):
        inst = self.opcodes.get(self.instr, lambda *a:self.nop())
        return inst(self)

    def _debug_exec(self):
        inst = self.opcodes.get(self.instr, lambda *a:self.nop())
        # more debugging to be added
        print("executing instruction:", inst.__name__, 'before pc', self.pc)
        inst(self)
        print("after pc", self.pc)


    def run(self, max_pc = -1):
        while not (self.sr & self.HALTING_BIT):  # Check if the halting bit is set
            instruction = self.fetch_instruction(1)
            # execute current instruction
            self.exec()
            if max_pc > 0 and self.pc > max_pc:
                break

    def interrupt(self, int):
        handler = self.interrupt_table.get(int, 0x0000)



