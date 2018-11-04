import os,sys,glob,cv2,subprocess
import pprint 

if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    diffCommandTemplate = "diff {} {}"
    simList = {}
    for fn1 in glob.glob("save_blob_list_image*"):
        for fn2 in glob.glob("save_blob_list_image*"):
            setDiff = diffCommandTemplate.format(fn1,fn2)
            diffProc = subprocess.Popen(setDiff.split(' '),stdout=subprocess.PIPE)
            output_b,isSuccess = diffProc.communicate()
            if output_b == '' and fn1 != fn2:
                if fn1 not in simList.keys():
                    simList[fn1] = []
                simList[fn1].append(fn2)
    pp.pprint(simList)
