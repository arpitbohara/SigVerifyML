
import os
import matplotlib.image as mpimg

from SigVerAPI.extractFeature import getFeatures


def getCSVFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    temp = getFeatures(path, display=display)
    features = (temp[0], temp[1][0], temp[1][1], temp[2], temp[3], temp[4][0], temp[4][1], temp[5][0], temp[5][1])
    return features

def makeCSV(genuine_image_paths,forged_image_paths,start=1,end=14):
    features_base_folder=os.getenv('IMAGE_FEATURES_BASE')
    train_feature_folder=os.getenv('TRAINING_FEATURE_FOLDER')
    test_feature_folder=os.getenv('TESTING_FEATURE_FOLDER')
    if not(os.path.exists(features_base_folder)):
        os.mkdir(features_base_folder)
        print('New folder "Features" created')
    if not(os.path.exists(train_feature_folder)):
        os.mkdir(train_feature_folder)
        print('New folder "Features/Training" created')
    if not(os.path.exists(test_feature_folder)):
        os.mkdir(test_feature_folder)
        print('New folder "Features/Testing" created')
    # genuine signatures path
    gpath = genuine_image_paths
    # forged signatures path
    fpath = forged_image_paths
    for person in range(start,end):
        per = ('00'+str(person))[-3:]
        print('Saving features for person id-',per)
        
        with open(train_feature_folder+'/training_'+per+'.csv', 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Training set
            for i in range(0,3):
                source = os.path.join(gpath, per+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(0,3):
                source = os.path.join(fpath, '021'+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')
        
        with open(test_feature_folder+'/testing_'+per+'.csv', 'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Testing set
            for i in range(3, 5):
                source = os.path.join(gpath, per+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',1\n')
            for i in range(3,5):
                source = os.path.join(fpath, '021'+per+'_00'+str(i)+'.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features))+',0\n')