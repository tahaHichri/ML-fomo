import os
import sys
import subprocess

def main():
    # where am I?
    location = os.path.dirname(os.path.realpath(__file__))
    
    # try to install
    try:
        os.system("pip install -r " + location + "/dependencies.txt")
    except Exception as e:
        print("Error Occured: " + str(e))
        return


if __name__ == '__main__':
    main()