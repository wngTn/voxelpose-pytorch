import pickle
import numpy as np
import os
import shutil
import json

'''
This file converts the output of voxelpose to the output of easymocap
It tracks the people through easy euclidian distance between each frame
'''


def prepare_out_dirs(prefix='output_easymocap/', dataDir='keypoints3d'):
    output_dir = os.path.join(prefix, dataDir)
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def convToEasyMocap(preds):
    output_dir = prepare_out_dirs()
    for j, frame in enumerate(preds):
        frame_num = '{:06d}'.format(j)
        file_name = frame_num + '.json'
        file_path= os.path.join(output_dir, file_name)
        for pred in frame:
                pred['id'] = int(pred['id'])
                pred['keypoints3d'] = pred['keypoints3d'] / 1000
                pred['keypoints3d'] = [[round(c, 3) for c in row] for row in pred['keypoints3d'].tolist()]

        json_string = json.dumps(frame)
        with open(file_path, 'w') as outfile:
            outfile.write(json_string)
            print('Saved:', file_path)



def main():
    res = []
    with open('output_vis/pred_voxelpose.pkl', 'rb') as f:
        preds = pickle.load(f)
        first_frame = preds[2000]

        # list of the positions of all the people in the first frame
        # WARNING: For this all people must appear in the first frame
        people = {}
        for i in range(len(first_frame)):
            people[i] = first_frame[i][:, :-2]

        # we start with second frame
        for i, (k, v) in enumerate(preds.items()):
            dist_matrix = []
            for pred in v:
                # only the 3 coordinates
                pred = pred[:, :-2]
                # calculate distance with first person, second person and so on
                # dist_row = [dist_to_id_0, dist_to_id_1, dist_to_id_2]
                dist_row = []
                for k, predecessor in people.items():
                    dist_row.append(np.sum(np.linalg.norm(pred - predecessor)))

                dist_matrix.append(dist_row)
            
            # dist_matrix = [
            # [dist_first_person_to_id_0...]
            # [dist_second_person_to_id_0...]
            # ]
            dist_matrix = np.array(dist_matrix)
            res_temp_list = []
            for l in range(len(dist_matrix)):
                ind = dist_matrix[:, l].argmin()
                res_temp = {
                    'id': l,
                    'keypoints3d': v[ind][:, :-2]
                }
                res_temp_list.append(res_temp)
                people[l] = v[ind][:, :-2]

            res.append(res_temp_list)
        convToEasyMocap(res)









if __name__=='__main__':
    main()