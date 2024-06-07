import cv2
import numpy as np


def cropBoard(frame_orig):
    frame = frame_orig.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_gray = cv2.GaussianBlur(frame_gray, ksize = (5, 5), sigmaX=1)

    frame_gray = cv2.Canny(frame_gray, threshold1=50, threshold2=125)

    cntrs, h = cv2.findContours(frame_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []

    for cnt in cntrs:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*perimeter, closed = True)
        bbox = cv2.boundingRect(approx)

        cnts.append((cnt, approx, area, perimeter, bbox))

    cnts = sorted(cnts, key = lambda x: -x[2])
    

    cnt = [c[0] for c in cnts[:1]]
    bbox = [c[4] for c in cnts[:1]]
    x, y, w, h = bbox[0]

    frame = frame[y:y+h, x:x+w]

    frame = cv2.resize(frame, (1000, 1000))
 

    return frame

def move_ellipse_center(ellipse, area):
    (cx, cy), (ma, Ma), angle = ellipse
    if area <= 800:
        return int(cx), int(cy)
    elif 750 < area < 2500:
        minor_axis_length = min(ma, Ma)
        angle_radians = np.radians(angle)
        dx = np.cos(angle_radians)
        dy = np.sin(angle_radians)
        new_cx = cx - dx * minor_axis_length 
        new_cy = cy - dy * minor_axis_length - (area - 500) / 30
        return int(new_cx), int(new_cy)
    else:
        major_axis_length = max(ma, Ma)
        angle_radians = np.radians(angle)
        dx = np.sin(angle_radians)
        dy = np.cos(angle_radians)
        new_cx = cx - abs(dx * major_axis_length)  
        new_cy = cy - abs(dy * major_axis_length) 
        return int(new_cx), int(new_cy)
        



def getDartPoints(image, draw = False):
    # draw_image = image.copy()
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for red color (Red can span two ranges in HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Define HSV range for green color
    lower_green = np.array([36, 70, 50])
    upper_green = np.array([86, 255, 255])

    # Create masks for red and green colors
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    kernel = np.ones((7, 7), np.uint8)
    mask_red = cv2.erode(mask_red, kernel, iterations=2)
    mask_green = cv2.erode(mask_green, kernel, iterations=1)

    # Find contours in the masks
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_red = [cnt for cnt in contours_red if cv2.contourArea(cnt) >= 125]
    contours_green = [cnt for cnt in contours_green if cv2.contourArea(cnt) >= 125]

    # Create copies of the masks to draw contours on
    # mask_red_with_contours = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR)
    # mask_green_with_contours = cv2.cvtColor(mask_green, cv2.COLOR_GRAY2BGR)

    green_points = []
    red_points = []

    for contour in contours_red:
        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)

        area = cv2.contourArea(contour)
        center = move_ellipse_center(ellipse, area)
        red_points.append(center)
        # if draw:
        #     cv2.drawContours(mask_red_with_contours, [contour], -1, (0, 0, 255), 2)
        #     cv2.ellipse(mask_red_with_contours, ellipse, (0, 0, 255), 2)
        #     cv2.circle(draw_image, center, 10, (0, 0, 255), -1)
        #     cv2.putText(mask_red_with_contours, f'{area:.2f}', tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for contour in contours_green:
        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)

        area = cv2.contourArea(contour)
        center = move_ellipse_center(ellipse, area)
        green_points.append(center)

        # if draw:
        #     cv2.drawContours(mask_green_with_contours, [contour], -1, (0, 255, 0), 2)
        #     cv2.ellipse(mask_green_with_contours, ellipse, (0, 255, 0), 2)
        #     cv2.circle(draw_image, center, 10, (0, 255, 0), -1)
        #     cv2.putText(mask_green_with_contours, f'{area:.2f}', tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return {'green':green_points, 'red':red_points}


def draw_points(image, points_dict):
    draw_img = image.copy()
    color_bgr__dict = {'red': (0, 0, 255), 'green': (0, 255, 0)}
    for team, pts in zip(points_dict.keys(), points_dict.values()):
        color = color_bgr__dict[team]
        for point in pts:
            cv2.circle(draw_img, point, 7, color, -1)

    return draw_img


def find_center(image):
    w, h = image.shape[0] // 2, image.shape[1] // 2
    cv2.circle(image, (w, h), 7, (255, 0, 255), -1)
    return (w, h + 10)


def count_scores(image, center, points_dict):
    draw_img = image.copy()
    color_bgr__dict = {'red': (0, 0, 255), 'green': (0, 255, 0)}
    def dist_to_score(dist):
        ratio_list = [4, 5, 6, 5, 6, 5, 6, 6]
        scores = [100, 80, 60, 50, 40, 30, 20, 10]
        koef = (image.shape[0] // 2) / sum(ratio_list)
        cur_pos = 0
        i = 0
        while True:
            if i >= len(ratio_list) - 1: return 0
            
            l = cur_pos * koef
            u = (cur_pos + ratio_list[i]) * koef
            if l < dist < u:
                return scores[i]
            else:
                cur_pos += ratio_list[i]
                i += 1
            

    team_scores_dict = {}
    for team, pts in points_dict.items():
        color = color_bgr__dict[team]
        team_scores_dict[team] = 0
        for point in pts:
            dist = np.linalg.norm(np.array(center) - np.array(point))
            team_scores_dict[team] += dist_to_score(dist)
            cv2.line(draw_img, center, point, color, 3, -1)

    return draw_img, team_scores_dict


def putResults(draw_image, scores_dict):
    color_bgr__dict = {'red': (0, 0, 255), 'green': (0, 255, 0)}

    x = y = 50

    for team, score in scores_dict.items():
        text = f"{team}: {score}"
        color = color_bgr__dict[team]
        cv2.putText(draw_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
        y += 50

    return draw_image


def process_image(img_path):
    img = cv2.imread(img_path)

    w, h = img.shape[1]//3, img.shape[0]//3

    img = cv2.resize(img, (w, h))

    img = cropBoard(img)

    points_dict = getDartPoints(img)
    draw_image = draw_points(img, points_dict)
    center = find_center(draw_image)
    draw_image, scores_dict = count_scores(draw_image, center, points_dict)

    draw_image = putResults(draw_image, scores_dict)

    return img, draw_image, scores_dict



def main():
    path1 = r'imgs/IMG_20240510_172748.jpg'
    path2 = r'imgs/IMG_20240510_172837.jpg'
    path3 = r'imgs/IMG_20240510_172930.jpg'

    img, draw_image, scores_dict = process_image(path3)

    cv2.imshow('Original', img)
    cv2.imshow('Points', draw_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)


if __name__ == '__main__':
    main()