import subprocess

def runCommandProcess(setCommand,dontBlock=False):
    modelProc = subprocess.Popen(setCommand.split(' '),stdout=subprocess.PIPE)
    if dontBlock:
        return ''
    output_b,isSuccess = modelProc.communicate()
    assert isSuccess is None, "ERROR; command failed."
    output = output_b.decode('utf-8')
    return output
