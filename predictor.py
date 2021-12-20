import sys

def list_command(args):
    print('invoked list with args: ' + str(args))

def version_command(args):
    print('invoked version with args: ' + str(args))

def help_command(args):
    print('invoked help with args: ' + str(args))

def apply_command(args):
    print('invoked apply with args: ' + str(args))

commands = {'list': list_command, 'version': version_command, 'help': help_command, 'apply': apply_command}

if len(sys.argv) == 1:
    print('Usage: ' + sys.argv[0] + ' command [options]')
    print('commands: ')
    print('  list:    list all available models.')
    print('  apply:   apply model to an example')
    print('  help:    print this help and exit.')
    print('  version: print version of the program and exit.')
else:
    command = sys.argv[1]
    arguments = sys.argv[2:]  
    if (command in commands):
        commands[command](arguments)
    else:
        print('command ' + command + ' is not supported')
