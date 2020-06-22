
# python3 app.py -t dl -m ../models/custom_cnn_2d.h5 -s ../data/samples/iza_meu_talisma.mp3   (실행하는 방법)
# python3 app.py  -s music/jazz.mp3
# 소스 수정하는 부분 :  ../gtzan/__init__.py


# https://github.com/Hguimaraes/gtzan.keras


import argparse
from gtzan import AppManager

# Constants
genres = {
    'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
    'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9
}

# @RUN: Main function to call the appmanager


def main(args):
    if args.type not in ["dl", "ml"]:
        raise ValueError(
            "Invalid type for the application. You should use dl or ml.")

    app = AppManager(args, genres)
    app.run()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Music Genre Recognition on GTZAN')

    # Required arguments
    parser.add_argument('-t', '--type', help='dl', type=str, default='dl')

    # Nearly optional arguments. Should be filled according to the option of the requireds
    parser.add_argument('-m', '--model', help='Path to trained model',
                        type=str, default='models/custom_cnn_2d.h5')
    parser.add_argument('-s', '--song', help='Path to song to classify',
                        type=str, default='music/sample.mp3')
    args = parser.parse_args()

    # Call the main function
    main(args)
