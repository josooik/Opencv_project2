# Opencv 차선인식
import cv2
import pickle
import numpy as np
import argparse
import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


@torch.no_grad()

def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/mov',  # file/dir/URL/glob, 0 for webcam
        imgsz=1280,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'data/mov',  # save results to project/name
        name='result',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != '0'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)

    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != '0':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:  # tensorflow model (tflite, pb, saved_model)
            imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
            pred[..., 0] *= imgsz[1]  # x
            pred[..., 1] *= imgsz[0]  # y
            pred[..., 2] *= imgsz[1]  # w
            pred[..., 3] *= imgsz[0]  # h
            pred = torch.tensor(pred)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line =                                                                                                                                                                                               (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')

            # Save results (image with detections)
            if save_img:
                # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/mov/output.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'data/mov', help='save results to project/name')
    parser.add_argument('--name', default='result', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


nwindows = 9
margin = 150
minpix = 1

trap_bottom_width = 0.8
trap_top_width = 0.1
trap_height = 0.4

road_width = 2.5  # 도로 폭

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

capture = cv2.VideoCapture('data/mov/mov/mov5.mp4',)

play_mode = 1 # 0: play once 1:play continuously

if capture.isOpened() == False:
  print("카메라를 열 수 없습니다.")
  exit(1)

video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
codec = cv2.VideoWriter_fourcc('m','p','4','v') # .mp4
#codec = cv2.VideoWriter_fourcc('M','J','P','G') # .avi

fps = 30.0
# 동영상 파일을 저장하려면 VideoWrite객체를 생성
# VideoWriter객체를 초기화 하기 위해 저장할 동영상 파일 이름,
# 코덱, 프레임레이트, 이미지 크기를 지정해야함
writer = cv2.VideoWriter('data/mov/output.mp4', codec, fps, (width,height))
writer1 = cv2.VideoWriter('data/mov/process.mp4', codec, fps, (5120, 2880), isColor=True)

#VideoWriter객체를 성공적으로 초기화 했는지 체크
if writer.isOpened() == False:
   print('동영상 저장파일객체 생성하는데 실패하였습니다.')
   exit(1)

if writer1.isOpened() == False:
   print('동영상 저장파일객체 생성하는데 실패하였습니다.')
   exit(1)

#Esc키를 눌러 동영상을 중단하면 종료직전까지 동영상이 저장됨
video_counter = 0

if capture.isOpened() == False:
    print("동영상을 열수없습니다.")
    exit(1)

def undistort(img, cal_dir='data/mov/p/wide_dist_pickle.p'):
    # cv2.imwrite('camera_cal/test_cal.jpg', dst)
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst

if capture.isOpened() == False:
    print("동영상을 열수없습니다.")
    exit(1)

while True:
    ret, img_frame = capture.read()

    if img_frame is None:
        break

    img_frames = img_frame.copy()
    img_frames1 = img_frame.copy()
    img_frames2 = img_frame.copy()

    # 왜곡 보정
    #img_undist = undistort(img_frames)

    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    #img_gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
    img_grays = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # BGR -> HSL 변환
    img_hls = cv2.cvtColor(img_frames, cv2.COLOR_BGR2HLS)
    #img_hls = cv2.cvtColor(img_undist, cv2.COLOR_BGR2HLS)

    img_hls_h, img_hls_l, img_hls_s = cv2.split(img_hls)
    img_hls_hs = cv2.merge((img_hls_h, img_hls_h, img_hls_h))
    img_hls_ls = cv2.merge((img_hls_l, img_hls_l, img_hls_l))
    img_hls_ss = cv2.merge((img_hls_s, img_hls_s, img_hls_s))

    for sigma in range(1, 4):
        img_GaussianBlur = cv2.GaussianBlur(img_hls_l, (1, 1), sigma)
        #cv2.imshow('img_GaussianBlur', img_GaussianBlur)

    #  소벨 필터 적용
    img_sobel_x = cv2.Sobel(img_GaussianBlur, cv2.CV_64F, 1, 1)
    img_sobel_xs = cv2.merge((img_sobel_x, img_sobel_x, img_sobel_x))

    img_sobel_x_abs = abs(img_sobel_x)
    img_sobel_x_abss = cv2.merge((img_sobel_x_abs, img_sobel_x_abs, img_sobel_x_abs))

    img_sobel_scaled = np.uint8(img_sobel_x_abs * 255 / np.max(img_sobel_x_abs))
    img_sobel_scaleds = cv2.merge((img_sobel_scaled, img_sobel_scaled, img_sobel_scaled))

    sx_threshold = (15, 255)
    sx_binary = np.zeros_like(img_sobel_scaled)
    sx_binary[(img_sobel_scaled >= sx_threshold[0]) & (img_sobel_scaled <= sx_threshold[1])] = 255
    sx_binarys = cv2.merge((sx_binary, sx_binary, sx_binary))

    s_threshold = (100, 255)
    s_binary = np.zeros_like(img_hls_s)
    s_binary[(img_hls_s >= s_threshold[0]) & (img_hls_s <= s_threshold[1])] = 255
    s_binarys = cv2.merge((s_binary, s_binary, s_binary))

    img_binary_added = cv2.addWeighted(sx_binary, 1.0, s_binary, 1.0, 0)
    img_binary_addeds = cv2.merge((img_binary_added, img_binary_added, img_binary_added))

    height, width = img_binary_added.shape[:2]

    dst_size = (width, height)
    src = np.float32([(0.42, 0.65), (0.52, 0.65), (0.1, 1), (0.8, 1)])
    src = src * np.float32((width, height))

    dst = np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])
    dst = dst * np.float32(dst_size)

    M = cv2.getPerspectiveTransform(src, dst)
    img_warp = cv2.warpPerspective(img_binary_added, M, dst_size)
    img_warps = cv2.merge((img_warp, img_warp, img_warp))

    img_warp1 = cv2.warpPerspective(img_frames1, M, dst_size)

    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)

    # axis=0->x축 즉 x축의 모든 값을 sum(더한다)는 의미
    histogram = np.sum(img_warp[height // 2:, :], axis=0)

    # axis=0->x축 즉 x축의 모든 값을 sum(더한다)는 의미
    midpoint = len(histogram) // 2

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = int(height / nwindows)

    # 영상에서 0이 앙닌 모든 점의 x,y점을 정의
    nonzero = img_warp.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # 왼쪽, 오른쪽 차선의 nonzero index를 받기위해 리스트 생성
    left_lane_inds = []
    right_lane_inds = []

    leftx_current = leftx_base
    rightx_current = rightx_base

    for window in range(nwindows):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(img_warp1, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (100, 255, 255), 3)
        cv2.rectangle(img_warp1, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (100, 255, 255), 3)

        # 아래 조건을 만족하는 점들의 인덱스 값을 리턴함
        good_left_inds = ((nonzero_y >= win_y_low) &
                          (nonzero_y < win_y_high) &
                          (nonzero_x >= win_xleft_low) &
                          (nonzero_x < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzero_y >= win_y_low) &
                           (nonzero_y < win_y_high) &
                           (nonzero_x >= win_xright_low) &
                           (nonzero_x < win_xright_high)).nonzero()[0]

        # 리스트에 조건을 만족하는 인덱스 값을 append
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # if you found > minpix 픽셀의 개수가 minpix보다 크면 사각형의 센터값 업데이트
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzero_x[good_left_inds]))

        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzero_x[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzero_x[left_lane_inds]
    lefty = nonzero_y[left_lane_inds]

    rightx = nonzero_x[right_lane_inds]
    righty = nonzero_y[right_lane_inds]

    # 곡선이므로 2차방정식 기준에 의해 계수들 구함
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    # 10개 데이터에 대해 평균값을 이용함으로써 중간의 튀는 값을 막음
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])

    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    # x와 y값을 그리기 위해 생성
    # 0부터 height-1(99)까지 height(100)개 만큼 1차원 배열 만들기
    ploty = np.linspace(0, height - 1, height)

    left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
    right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

    # 선에 색칠하기
    img_warp1[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 100]
    img_warp1[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [255, 0, 100]

    color_img = np.zeros_like(img_warp1)
    left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    points = np.hstack((left, right))

    mid = (left_fitx + right_fitx) // 2

    left_fix_mean = np.mean(left_fitx)
    right_fix_mean = np.mean(right_fitx)

    mid_points = np.array([np.transpose(np.vstack((mid, ploty)))])

    mean_mid = np.mean(mid)

    # 차선 그리기
    #cv2.polylines(color_img, np.int_(points), False, (0, 255, 255), 10)

    # 차선 안쪽 채우기
    cv2.fillPoly(color_img, np.int_(points), (0, 255, 0))

    # 원본영상과 차선검출 영상 합치기 : 가중치 조절(1:100%, 0.4:40%)
    img_warp2 = cv2.addWeighted(img_warp1, 1, color_img, 0.5, 0)

    cv2.line(img_warp2, (width // 2 - 60, 700), (width // 2 - 60, 940), (0, 0, 255), 10)
    #cv2.line(img_warp2, (width // 2, 0), (width // 2, height), (0, 0, 255), 20)
    cv2.line(img_warp2, (np.int_(mean_mid), 800), (np.int_(mean_mid), 1000), (255, 0, 0), 20)
    #cv2.polylines(img_warp2, np.int_(mid_points), False, (255, 255, 128), 20)
    cv2.circle(img_warp2, (np.int_(mean_mid), height // 2 + 300), 10, (255, 0, 255), 20)
    cv2.line(img_warp2, (width // 2, height // 2 + 300), (np.int_(mean_mid), height // 2 + 300), (255, 255, 255), 20)
    #cv2.imshow('img_frames2', img_frames2)

    # inverse버전
    M1 = cv2.getPerspectiveTransform(dst, src)  # src: 4개의 원본 좌표점  dst: 4개의 결과 좌표점 '

    # src의 좌표점 4개를 dst의 좌표점으로 인식한다. 저 경우에는 각 점을 줄이겠다는 의미(그만큼 축소)
    img_warp3 = cv2.warpPerspective(img_warp2, M1, dst_size)

    img_line = cv2.addWeighted(img_frames2, 1, img_warp3, 0.5, 0)
    img_f = cv2.addWeighted(img_frames1, 1, img_line, 0.5, 0)

    road_width_pixel = road_width / (right_fix_mean - left_fix_mean) # pixel 1 = m
    error = width // 2 - mean_mid # 도로중심과 이미지 중심이 떨어진 픽셀 거리
    dis_error = error * road_width_pixel # 도로중심과 이미지 중심이 떨어진 거리

    if(dis_error < 0):
        cv2.putText(img_f, 'right : %.2fm' %(abs(dis_error)), (770, 750), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    elif (dis_error > 0):
        cv2.putText(img_f, 'left : %.2fm' %(abs(dis_error)), (770, 750), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    else:
        cv2.putText(img_f, 'center : %.2fm' %(abs(dis_error)), (770, 750), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cont = cv2.hconcat([img_frame, img_frame, img_grays, img_hls])
    cont1 = cv2.hconcat([img_hls_hs, img_hls_ls, img_hls_ss, img_sobel_scaleds])
    cont2 = cv2.hconcat([img_sobel_xs, img_sobel_x_abss])
    cont3 = cv2.hconcat([sx_binarys, s_binarys, img_binary_addeds, img_warps])
    cont4 = cv2.hconcat([img_warp1, img_warp2, img_warp3, img_f])
    cont5 = cv2.vconcat([cont, cont1, cont3, cont4])

    writer.write(img_f)
    writer1.write(cont5)

    cont5 = cv2.pyrDown(cont5)
    cont5 = cv2.pyrDown(cont5)
    cont5 = cv2.pyrDown(cont5)
    #imgs = cv2.pyrDown(imgs)
    img_f = cv2.pyrDown(img_f)
    #img_warp2 = cv2.pyrDown(img_warp2)
    #cv2.imshow('img_warp2', img_warp2)
    cv2.imshow('cont5', cont5)
    cv2.imshow('img_f', img_f)

    imgs1 = cv2.pyrDown(cont2)
    imgs1 = cv2.pyrDown(imgs1)
    #cv2.imshow('imgss1', imgs1)

    key = cv2.waitKey(1)
    if key == 27:  # ESC 키
        break

    if video_counter == video_length:
        video_counter = 0
    else:
        video_counter += 1

    # 동영상이 끝나면 재생되는 프레임의 위치를 0으로 다시 지정
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

capture.release()
writer.release()
writer1.release()
cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    run(**vars(opt))
