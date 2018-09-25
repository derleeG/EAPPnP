import cv2
import numpy as np
import EAPPnP
import stochastic.continuous
import skvideo.io
import scipy.spatial
import matplotlib.pyplot as plt


REF = []
for i in [-1, 1]:
    for j in [-1, 1]:
        for k in [-1, 1]:
            REF.append((i, j, k))
REF = np.array(REF, dtype=np.float32)
LINKS = []
for idx1, v1 in enumerate(REF):
    for idx2, v2 in enumerate(REF[idx1+1:]):
        if np.sum(v1 == v2) == 2:
            LINKS.append((idx1, idx1+1+idx2))


def OBBIOU(OBB1, OBB2):

    I = OBBintersection(OBB1, OBB2)
    _, _, S1 = OBB1
    _, _, S2 = OBB2

    U = 8*np.prod(S1) + 8*np.prod(S2) - I

    return I/U if I != 0 else 0



def OBBintersection(OBB1, OBB2):
    R1, T1, S1 = OBB1
    R2, T2, S2 = OBB2

    B1 = np.matmul(REF*S1, R1.T) + T1
    B2 = np.matmul(REF*S2, R2.T) + T2

    F1 = [(R1[:, 0], B1[-1, :]),
          (R1[:, 1], B1[-1, :]),
          (R1[:, 2], B1[-1, :]),
          (-R1[:, 0], B1[0, :]),
          (-R1[:, 1], B1[0, :]),
          (-R1[:, 2], B1[0, :])]
    F2 = [(R2[:, 0], B2[-1, :]),
          (R2[:, 1], B2[-1, :]),
          (R2[:, 2], B2[-1, :]),
          (-R2[:, 0], B2[0, :]),
          (-R2[:, 1], B2[0, :]),
          (-R2[:, 2], B2[0, :])]

    P = []

    for B, F in [(B1, F2), (B2, F1)]:
        for i, j in LINKS:
            p0, p1 = B[i, :], B[j, :]
            te, tl = [], []
            for n, v in F:
                t = np.dot(n, v-p0)/np.dot(n, p1-p0)
                if np.dot(n, p1 - p0) > 0:
                    tl.append(t)
                else:
                    te.append(t)
            te = max(0, max(te))
            tl = min(1, min(tl))

            if tl >= te:
                P.append(p0 + te*(p1-p0))
                P.append(p0 + tl*(p1-p0))
    if P:
        P = np.stack(P)
        H = scipy.spatial.ConvexHull(P)

        return H.volume
    else:
        return 0



def gen_point_on_box(n, mode=0):
    x = np.linspace(-1, 1, n)
    P = []
    for i in x:
        for j in x:
            for k in x:
                if np.sum([np.abs(m) == 1 for m in [i,j,k]]) >= mode:
                    P.append([i,j,k])

    P = np.array(P, dtype=np.float32)
    return P

def gen_point_on_box2(n):
    P = np.random.rand(n, 3).astype(np.float32)*2 - 1
    return P

def correlated_uniform_generator(limit=(-1, 1)):
    # not uniform not all
    # using Fractional Brownian Motion with wrap around

    FBM = stochastic.continuous.FractionalBrownianMotion(0.90, 20)
    x0, sign, offset = 0, 1, 0

    while True:
        noises = FBM.sample(2000)

        for noise in noises:
            x = (x0 + noise)*sign + offset

            # check boundary
            while x > limit[1] or x < limit[0]:
                if x > limit[1]:
                    sign, offset = -sign, -offset + 2*limit[1]
                    x = (x0 + noise)*sign + offset
                if x < limit[0]:
                    sign, offset = -sign, -offset + 2*limit[0]
                    x = (x0 + noise)*sign + offset

            yield x
        x0 = x


def uniform_generator(limit=(-1, 1)):

    while True:
        noises = np.random.rand(2000)*(limit[1] - limit[0]) + limit[0]

        for noise in noises:
            yield noise


def RTS_state_generator(correlated=True):
    # parameters for the random motion
    limits = [(-2, 2), (-2, 2), (-2, 2), (-4, 4), (-3, 3), (0.4, 5), (-1, 1), (-1, 1)]

    if correlated:
        rngs = [correlated_uniform_generator(limit) for limit in limits]
    else:
        rngs = [uniform_generator(limit) for limit in limits]

    for state in zip(*rngs):
        r = cv2.Rodrigues(np.array(state[:3]))[0].astype(np.float32)
        t = np.array(state[3:6], dtype=np.float32)
        t[-1] *= 15
        s = np.power(2, np.array([0, *state[6:]])).astype(np.float32)

        yield r, t, s


def perspective_project(x):
    x = x[:, :2]/x[:, 2].reshape(-1, 1)
    return x


def draw_perspective_point(p, f, img, color):

    y, x, _ = img.shape
    imp = p*f + np.array([x/2, y/2])
    for x, y in imp.astype(np.int32):
        cv2.circle(img, (x, y), 2, color, -1, cv2.LINE_AA)

    return


def draw_perspective_box(box, f, img, color):

    y, x, _ = img.shape
    p = box.reshape(2, 2)
    imp = p*f + np.array([x/2, y/2])
    x1, y1, x2, y2 = imp.reshape(-1).astype(np.int32)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, ' (W: {}, H: {})'.format(x2-x1, y2-y1), (x2, y2),
            font, 0.4, color, 1, cv2.LINE_AA)

    return


def find_2D_bounding_box(p):
    pmin = np.amin(p, 0)
    pmax = np.amax(p, 0)
    pmin, pmax = pmin*1.1 - pmax*0.1, pmax*1.1 - pmin*0.1
    box = np.concatenate((pmin, pmax))

    return box


def orthogonal_project(x, dim):
    dims = [x for x in range(3) if x != dim]
    x = x[:, dims]
    return x


def gen_orthogonal_view_setting(p, dim, show_size):

    p = orthogonal_project(p, dim)
    box = find_2D_bounding_box(p).reshape(2, 2)
    center = np.mean(box, 0)
    scale = show_size / np.max(box[1, :] - box[0, :])

    return dim, center, scale


def draw_orthogonal_point(p, view_setting, img, color):
    dim, center, scale = view_setting
    y, x, _ = img.shape
    p = (orthogonal_project(p, dim)-center)*scale + np.array([x/2, y/2])

    for x, y in p.astype(np.int32):
        cv2.circle(img, (x, y), 2, color, -1, cv2.LINE_AA)

    return p


def gen_orthogonal_view(OBB_gt, OBB, X, image_size, dim):
    R_gt, T_gt, S_gt = OBB_gt
    R, T, S = OBB

    Y_gt = np.matmul(X*S_gt, R_gt.T) + T_gt
    Y = np.matmul(X*S, R.T) + T

    B_gt = np.matmul(REF*S_gt, R_gt.T) + T_gt
    B = np.matmul(REF*S, R.T) + T

    view_setting = gen_orthogonal_view_setting(\
            B_gt, dim, image_size[0]/2)
    view = np.zeros((*image_size, 3), dtype=np.uint8)
    draw_orthogonal_point(Y_gt, view_setting, view, (255, 255, 255))
    draw_orthogonal_point(Y, view_setting, view, (100, 100, 255))
    b_gt = draw_orthogonal_point(B_gt, view_setting, view, (255, 255, 255))
    b = draw_orthogonal_point(B, view_setting, view, (100, 100, 255))

    b_gt = b_gt.astype(np.int32)
    b = b.astype(np.int32)
    for i, j in LINKS:
        cv2.line(view, (b_gt[i, 0], b_gt[i, 1]), (b_gt[j, 0], b_gt[j, 1]), (150, 150, 150), 1)
        cv2.line(view, (b[i, 0], b[i, 1]), (b[j, 0], b[j, 1]), (60, 60, 150), 1)

    view[:,0,:] = 255
    view[0,:,:] = 255
    view[:,-1,:] = 255
    view[-1,:,:] = 255

    return view


def draw_result_list(results, view):
    for idx, (roi_res, point_set, radius, rot_err, trans_err, iou, recall) \
            in enumerate(results):
        color = (255, 255, 255) if idx != len(results)-1 else (100, 100, 255)
        cv2.putText(view,'ROI RES: {0}x{0}, # POINTs: {1}, RADIUS: {2}'\
                .format(roi_res, point_set, radius),\
                (15, 60+45*idx), font, 0.4, color,1,cv2.LINE_AA)
        cv2.putText(view,'    ROT ERR: {:.3f} deg, TRANS ERR: {:.3f} %'.format(rot_err, trans_err),\
                (15, 75+45*idx), font, 0.4, color,1,cv2.LINE_AA)
        cv2.putText(view,'    IOU: {:.3f}, Recall0.7: {:.3f} %'.format(iou, recall),\
                (15, 90+45*idx), font, 0.4, color,1,cv2.LINE_AA)

    return view


def cal_rot_err(R_gt, R):
    err = []
    for r_gt, r in zip(R_gt, R):
        r_gt = r_gt/np.linalg.norm(r_gt)
        r = r/np.linalg.norm(r)
        err.append(np.arccos(np.clip(np.dot(r_gt, r), -1.0, 1.0)))

    return max(err)


def float_to_mask(p, r, n):
    # nxn size grid
    # True if d((i, j), p) < r
    rr = r*r
    x, y = np.meshgrid(range(n), range(n))
    x = x.reshape(1, n, n)
    y = y.reshape(1, n, n)
    px = p[:, 0].reshape(-1, 1, 1)
    py = p[:, 1].reshape(-1, 1, 1)

    mask = ((px - x)**2 + (py - y)**2 < rr)
    return mask


def mask_to_float(mask):

    n, m = mask.shape[-1], mask.shape[-2]
    x, y = np.meshgrid(range(n), range(m))
    mask = mask > 0.5
    p = []
    for m in mask:
        px = np.sum(x[m]) / np.sum(m)
        py = np.sum(y[m]) / np.sum(m)
        p.append((px, py))

    return np.array(p, dtype=np.float32)


def float_to_float(p, r, n):
    rr = r*r
    x, y = np.meshgrid(range(n), range(n))
    x = x.reshape(1, n, n)
    y = y.reshape(1, n, n)
    px = p[:, 0].reshape(-1, 1, 1)
    py = p[:, 1].reshape(-1, 1, 1)

    mask = ((px - x)**2 + (py - y)**2 < rr)

    px = np.sum((mask*x).reshape(-1, n*n), -1)/np.sum(mask.reshape(-1, n*n), -1)
    py = np.sum((mask*y).reshape(-1, n*n), -1)/np.sum(mask.reshape(-1, n*n), -1)

    p = np.stack((px, py), 1)
    return p


def roi_discretize(p, box, res):
    box = box.reshape(2, 2)
    scale = box[1, :] - box[0, :]
    offset = box[0, :]

    p = (np.round((p - offset)/scale*res - 0.5) + 0.5)/res*scale + offset

    return p


def roi_discretize2(p, box, res, r):
    box = box.reshape(2, 2)
    scale = box[1, :] - box[0, :]
    offset = box[0, :]

    p = (p - offset)/scale*res - 0.5
    p_o = p.copy()
    p = float_to_float(p, r, res)
    p = (p + 0.5)/res*scale + offset

    return p


if __name__ == '__main__':

    # build reference P
    image_size = (360, 640) # 1/3 of full HD
    image_size = (540, 960) # 1/3 of full HD
    f = 712 # approximately the same field of view as Iphone 6
    func = EAPPnP.EAPPnP
    visualize = False
    write_video = False

    rng = RTS_state_generator(correlated=visualize)

    # (roi_res, point_set)
    exp_list = [(56, 11), (56, 9), (56, 7), (56, 4), (56, 3), (56, 2),
                (7, 5), (14, 5), (28, 5), (56, 5), (112, 5), (224, 5)]
    exp_list = []

    #for a in [4.88, 0]:
    for a in np.linspace(0.71, 6.5, 1000):
        for m in [1]:
            for r in [56]:
                for n in [5]:
                    temp = gen_point_on_box(n, m)
                    n2 = (temp.shape[0] + 16)//12
                    exp_list.append((r, n2, 2, a))
                    #exp_list.append((r, n, 1, a))
    result_list = []


    if write_video:
        writer = skvideo.io.FFmpegWriter('result.mp4', {'-r': '60'}, {'-r': '60'})

    frame_count = 0
    while True:
        if frame_count % 3000 == 0:
            if result_list:
                r, n, rad, rot, trans, iou, recall = result_list[-1]
                print(('ROI res: {}, # points: {}, radius: {}, '
                       'rotation err: {:.3f}, translation err: {:.3f}, '
                       'IOU: {:.3f}, Recall0.7: {:.3f}')\
                               .format(r, n, rad, rot, trans, iou, recall))

            try:
                roi_res, point_set, gen_mode, radius = exp_list.pop()
            except:
                break
            X = gen_point_on_box(point_set, gen_mode)
            point_set = X.shape[0]

            rot_err_sum = 0
            trans_err_sum = 0
            IOU_sum = 0
            recall_sum = 0
            frame_count = 0
            result_list.append([roi_res, point_set, radius, 0, 0, 0, 0])

        R_gt, T_gt, S_gt = next(rng)
        Y_gt = np.matmul(X*S_gt, R_gt.T) + T_gt
        p_gt = perspective_project(Y_gt)
        box = find_2D_bounding_box(p_gt)

        if radius < 2**0.5/2:
            p_noise = roi_discretize(p_gt, box, roi_res)
        else:
            p_noise = roi_discretize2(p_gt, box, roi_res, radius)

        if func.__name__ == 'EAPPnP':
            R, T, S, _ = func(X, p_noise)
        else:
            R, T, _ = EAPPnP.EPPnP(X, p_noise)
            S = 1

        rot_err = cal_rot_err(R_gt, R)*180/np.pi
        trans_err = np.linalg.norm(T.reshape(-1)-T_gt)/np.linalg.norm(T)*100
        IOU = OBBIOU((R_gt, T_gt, S_gt), (R, T.T, S))

        rot_err_sum += rot_err
        trans_err_sum += trans_err
        IOU_sum += IOU
        if IOU > 0.7:
            recall_sum += 100
        frame_count += 1

        result_list[-1][-4] = rot_err_sum/frame_count
        result_list[-1][-3] = trans_err_sum/frame_count
        result_list[-1][-2] = IOU_sum/frame_count
        result_list[-1][-1] = recall_sum/frame_count

        Y = np.matmul(X*S, R.T) + T.reshape(-1)
        p = perspective_project(Y)

        if visualize:
            # visualize
            p_view = np.zeros((*image_size, 3), dtype=np.uint8)
            draw_perspective_point(p_gt, f, p_view, (255, 255, 255))
            draw_perspective_point(p_noise, f, p_view, (255, 100, 105))
            draw_perspective_point(p, f, p_view, (100, 100, 255))
            draw_perspective_box(box, f, p_view, (80, 80, 80))
            p_view[:,0,:] = 255
            p_view[0,:,:] = 255
            p_view[:,-1,:] = 255
            p_view[-1,:,:] = 255

            t_view = gen_orthogonal_view((R_gt, T_gt, S_gt), (R, T.T, S), X, image_size, 1)
            s_view = gen_orthogonal_view((R_gt, T_gt, S_gt), (R, T.T, S), X, image_size, 0)
            f_view = gen_orthogonal_view((R_gt, T_gt, S_gt), (R, T.T, S), X, image_size, 2)

            # add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(p_view,'Camera View',(15,30), font, 0.5,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Ground Truth',(15,60), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Discretized Ground Truth',\
                    (15,75), font, 0.4,(255,100,100),1,cv2.LINE_AA)
            cv2.putText(p_view,'Estimated',(15,90), font, 0.4,(100,100,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Image resolution: {}x{}'.format(image_size[1], image_size[0]),\
                    (15,105), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Focal length: {} pix/m'.format(f),\
                    (15,120), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'ROI resolution: {}x{}'.format(roi_res, roi_res),\
                    (15,135), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'# Points: {0}'.format(point_set),\
                    (15,150), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Cuboid size: {:.3f}x{:.3f}x{:.3f}'.format(S_gt[0]*2, S_gt[1]*2, S_gt[2]*2), \
                    (15,165), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Cuboid center: ({:.3f}, {:.3f}, {:.3f})'.format(T_gt[0], T_gt[1], T_gt[2]), \
                    (15,180), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Method: {}'.format(func.__name__),\
                    (15,195), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Rotation error: {:.3f} deg'.format(rot_err),\
                    (15,210), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Translation error: {:.3f} %'.format(trans_err),\
                    (15,225), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'IOU: {:.3f}'.format(IOU),\
                    (15,240), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Recall0.7: {:.3f} %'.format((IOU>0.7)*100),\
                    (15,255), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Average Rotation error: {:.3f} deg'.format(rot_err_sum/frame_count),\
                    (15,270), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Average Translation error: {:.3f} %'.format(trans_err_sum/frame_count),\
                    (15,285), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Average IOU: {:.3f}'.format(IOU_sum/frame_count),\
                    (15,300), font, 0.4,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(p_view,'Average Recall0.7: {:.3f} %'.format(recall_sum/frame_count),\
                    (15,315), font, 0.4,(255,255,255),1,cv2.LINE_AA)


            cv2.putText(t_view,'Top View',(15,30), font, 0.5,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(s_view,'Side View',(15,30), font, 0.5,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(f_view,'Front View',(15,30), font, 0.5,(255,255,255),1,cv2.LINE_AA)

            t_view = draw_result_list(result_list, t_view)
            vis = np.concatenate(
                    (np.concatenate((p_view, t_view), 1),
                    np.concatenate((s_view, f_view), 1)), 0)

            cv2.imshow('vis', vis)

            q = cv2.waitKey(10)
            if q == 27:
                break

            if write_video:
                writer.writeFrame(vis)

    result_list = np.array(result_list)

    plt.figure(1)
    plt.scatter(result_list[:, 2], result_list[:, -1])
    plt.xlabel('radius')
    plt.ylabel('recall(0.7)')
    plt.figure(2)
    plt.scatter(result_list[:, 2], result_list[:, -2])
    plt.xlabel('radius')
    plt.ylabel('IOU')
    plt.show()
