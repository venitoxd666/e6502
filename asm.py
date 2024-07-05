from e6502 import *

def encargspec(size, mem_reg):
    # mem_reg
    return {1:0, 2:1, 4:2,8:3}[size] | ((mem_reg  & 3) << 5)

class Assembler:
    def __init__(self):
        self.instructions = {
            'dps': b'\x80\x05',
            'add': b'\x9A',
            'sub': b'\x9B',
            'nop': b'\x00',
            'halt': b'\x40',
            'neg':b'\x90',
            'push':b"\x2B", 
            'pop':b'\x2A',
            'limn':b'\x1a',
            'mov':b'\x76'
        }
        self.register_page = RegisterPage()

    def mov(self, extend, args):
        args = args[1:]  # skip opcode (your code passed args with opcode, I optimized it but because your code broke deleting the opcode passing, then )
                         # args[0] is always None

        
        reg = self.register_page.register_from_prefix(args[0]) # fetch first argument which is always a register
        reg_size = self.register_page.register_size(reg)
        hard_one = args[1]
        machine_code = bytearray()
        machine_code.extend(self.instructions['mov'])

        if hard_one[0] == '*':
            # this is a pointer
            hard_one = hard_one[1:]
            if hard_one[0] == '0':
                # it's a literal
                decarg = 2 << 4
                hard_one = int(hard_one, base=16)
                arg = (hard_one.to_bytes(reg_size, 'little'))
            else:
                # it's a register
                decarg = 0
                hard_one = self.register_page.register_from_prefix(hard_one)
                arg = hard_one.to_bytes(1,'little')
        elif hard_one[0] == '%':
            hard_one = hard_one[1:]
            if hard_one[0] == '0':
                # it's a literal
                decarg = 3 << 4
                hard_one = int(hard_one, base=16)
                arg = (hard_one.to_bytes(2, 'little'))
            else:
                decarg = 1 << 4
                hard_one = self.register_page.register_from_prefix(hard_one)
                arg = hard_one.to_bytes(1,'little')
        else:
            decarg = 2 << 5 # set size to 2, this means inmediate is inmediate or register is register
            if hard_one[0] == '0':
                decarg += 2 << 4
                hard_one = int(hard_one, base = 16)
                arg = hard_one.to_bytes(reg_size, 'little')
            else:
                # print("this")
                hard_one = self.register_page.register_from_prefix(hard_one)
                arg = hard_one.to_bytes(1,'little')
        
        decarg += extend
        # print(hex(decarg))
        
        machine_code.append(decarg)
        machine_code.append(reg)
        machine_code.extend(arg)
        
        return machine_code
        

    def assemble(self, assembly_code):
        lines = assembly_code.splitlines()
        machine_code = bytearray()

        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):  # Skip empty lines and comments
                continue

            v = line.split(" ", 1)
            instr = v[0]
            parts = ""
            if len(v) - 1:
                instr, parts = v
            # print(instr, parts)
            args = parts.split(',')
            parts = [None, *map(lambda s:s.strip(), args)]
            # print(instr, parts)

            # if instr not in self.instructions:
                # raise ValueError(f"Unknown instruction: {instr}")

            if instr == 'dps':
                addr = int(parts[1], 16)  # Assuming the address is given in hexadecimal
                machine_code.extend(self.instructions[instr])
                machine_code.extend(addr.to_bytes(4, byteorder='little'))
            elif instr in {'add', 'sub'}:
                # print(parts)
                reg1 = self.register_page.register_from_prefix(parts[1].lower())
                reg2 = self.register_page.register_from_prefix(parts[2].lower())
                machine_code.extend(self.instructions[instr])
                machine_code.append(reg1)
                machine_code.append(reg2)
            elif instr in {'limn'}:
                reg = self.register_page.register_from_prefix(parts[1].lower())
                imm = int(parts[2], 16)  # Assuming the immediate is given in hexadecimal
                reg_size = self.register_page.register_size(reg)
                machine_code.extend(self.instructions[instr])
                machine_code.append(reg)
                machine_code.extend(imm.to_bytes(reg_size, byteorder='little'))
            elif instr in {'neg', 'pop', 'push'}: # put here al instructions that fit in INSTR {REG}:
                reg = self.register_page.register_from_prefix(parts[1].lower())
                
                machine_code.extend(self.instructions[instr])
                machine_code.append(reg)
            elif instr in {'mov', 'zmov'}:
                res = self.mov(0, parts)
                machine_code.extend(res)
            elif instr == 'smov':
                res = self.mov(1, parts)
                machine_code.extend(res)
            elif instr in {'nop', 'halt'}:
                machine_code.extend(self.instructions[instr])
            else:
                raise ValueError(f"Unsupported instruction: {instr}")

        return machine_code




