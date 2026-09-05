
from importlib import import_module
generator = import_module('MSC-STEP3A-GENERATE')
calibration = import_module('MSC-STEP3A-CALIBRATE')
stft = import_module('MSC-STEP3A-STFT')
cwt = import_module('MSC-STEP3A-CWT')
hilbert = import_module('MSC-STEP3A-HILBERT')
validation = import_module('MSC-STEP3A-VALIDATE')

def main():
    generator.generate()
    calibration.calibrate()
    stft.run()
    cwt.run()
    hilbert.run()
    validation.validate(validation.DEFAULT_TRUTH_FILE, [stft.OUTPUT_FILE, cwt.OUTPUT_FILE, hilbert.OUTPUT_FILE], validation.DEFAULT_OUTPUT_FOLDER)
if __name__ == '__main__':
    main()
