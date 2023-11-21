from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import traceback
from PIL import Image
from landmarkers import get_video_landmarks


app = Flask(__name__)
CORS(app)


@app.route('/file', methods=['POST'])
def get_file():
    if request.method == 'POST':
        try:
            f = request.files['file']
            f.save(secure_filename(f.filename))
            img1, backswing_frame, max2, hit_frame, hit_frame_after, similarity_right, is_left_up, last_cock_frames, is_hit_timing, is_elbow_ok, is_elbow_ok_after, is_wrist_used = get_video_landmarks(f.filename)
            img1 = Image.fromarray(img1)
            img1.save('result1.jpg', 'JPEG')
            backswing_frame = Image.fromarray(backswing_frame)
            backswing_frame.save('backswing_frame.jpg', 'JPEG')
            hit_frame = Image.fromarray(hit_frame)
            hit_frame.save('hit_frame.jpg', 'JPEG')
            hit_frame_after = Image.fromarray(hit_frame_after)
            hit_frame_after.save('hit_frame_after.jpg', 'JPEG')
            os.remove(f.filename)
            return jsonify({'result': 'ok',
                            'max2': max2,
                            'similarity_right': similarity_right,
                            'is_left_up': is_left_up,
                            'last_cock_frames': last_cock_frames,
                            'is_hit_timing': is_hit_timing,
                            'is_elbow_ok': is_elbow_ok,
                            'is_elbow_ok_after': is_elbow_ok_after,
                            'is_wrist_used': is_wrist_used
                            })
        except:
            print(traceback.print_exc())
            return jsonify({'result': 'error'})


if __name__ == '__main__':
    socketio.run(app)
    # app.run()

