from e6502 import *
from asm import Assembler
# from argparse import ArgumentParser
import argparse
import sys

def _flat(arr):
    n = []
    for o in arr:
        if isinstance(o, list):
            n.extend(_flat(o))
        else:
            n.append(o)    
    return n

def main():
    # argparse.
    parser = argparse.ArgumentParser(description="Assembler and Emulator for Custom CPU Architecture")
    # subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Assembler sub-command
    assemble_parser = parser # subparsers.add_parser('a', help="Assemble a file")
    assemble_parser.add_argument('-a', '--assemble', type=str, required=False, help="Path to the assembly file", default="")
    assemble_parser.add_argument('-o', '--output', type=str, required=False, help="Output file for the assembled binary", default="")

    # Emulator sub-command
    execute_parser = parser
    execute_parser.add_argument('-e', '--execute', type=str, required=False, help="Path to the binary file", default="")
    execute_parser.add_argument('-d', '--debug', action='store_true', help="Enable debug mode")
    execute_parser.add_argument('-p', '--print', type=str, nargs='+', help="Registers to print after execution")

    parser.add_argument('-op', '--opcode-table', action='store_true', dest='opcode_table', help="print opcode table and exit")

    args = parser.parse_args()
    if args.opcode_table:
        print_opcode_table(args.output)
    elif args.assemble:
        assert args.output
        assemble(args.assemble, args.output)
    elif args.execute:
        args.print = args.print or []
        execute(args.execute, args.debug, _flat([[l for l in r.split(',') if l] for r in args.print]))
    else:
        parser.print_help()
        
def assemble(ass_file, output):
    with open(ass_file, 'r') as f:
        code = f.read()
    
    bt = Assembler().assemble(code)
    
    with open(output, 'wb') as f:
        f.write(bt)

def print_opcode_table(output):
    backup = None
    # if an output is given, capture the stdout and redirect it to whatever's output is preferred
    if output:
        backup = sys.stdout
        sys.stdout = open(output, 'w')
    
    cpu = f8403()
    cpu.opcode_table()
    
    if output:
        sys.stdout.close()
        sys.stdout = backup

def execute(file, debug, registers):
    cpu = f8403()
    # print(registers)
    if debug:
        cpu.exec = cpu._debug_exec
    
    registers = list([(v, cpu.rp.register_from_prefix(v)) for v in registers])
    # would be a good idea to copy the stuff in ROM?
    # :p
    buff = cpu.ram
    with open(file, 'rb') as f:
        c = f.read(1)
        i = 0
        while(c != b''):
            buff[buff.start + i] = c[0]
            i += 1
            c = f.read(1)
    # print(cpu.ram._computed_memories)
    cpu.run()
    
    print("cpu halted after", cpu.cycle, "cycles")
    for reg in registers:
        print("register", reg[0], 'value is', hex(cpu.rp.get_register(reg[1])))

if __name__ == '__main__':
    main()