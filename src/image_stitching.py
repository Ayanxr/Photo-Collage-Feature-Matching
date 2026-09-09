import cv2
import numpy as np

from .feature_matching import (
    detect_features,
    match_features
)


def calculate_homography(
    image1,
    image2,
    keypoints1,
    keypoints2,
    matches
):
    """
    Calculate homography between two images
    using matched feature points and RANSAC.
    """

    if len(matches) < 4:
        raise ValueError(
            "Not enough matching features."
        )

    src_points = np.float32([
        keypoints2[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    dst_points = np.float32([
        keypoints1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        5.0
    )

    if H is None:
        raise ValueError(
            "Could not calculate homography."
        )

    return H, mask


def stitch_two_images(image1, image2):
    """
    Stitch image2 onto image1.
    """

    keypoints1, descriptors1 = detect_features(image1)
    keypoints2, descriptors2 = detect_features(image2)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError(
            "Could not detect enough features."
        )

    matches = match_features(
        descriptors1,
        descriptors2
    )

    if len(matches) < 4:
        raise ValueError(
            "Not enough good matches."
        )

    H, mask = calculate_homography(
        image1,
        image2,
        keypoints1,
        keypoints2,
        matches
    )

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    corners_image2 = np.float32([
        [0, 0],
        [w2, 0],
        [w2, h2],
        [0, h2]
    ]).reshape(-1, 1, 2)

    transformed_corners = cv2.perspectiveTransform(
        corners_image2,
        H
    )

    corners_image1 = np.float32([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ]).reshape(-1, 1, 2)

    all_corners = np.concatenate(
        (corners_image1, transformed_corners),
        axis=0
    )

    [xmin, ymin] = np.int32(
        all_corners.min(axis=0).ravel() - 0.5
    )

    [xmax, ymax] = np.int32(
        all_corners.max(axis=0).ravel() + 0.5
    )

    translation = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1]
    ])

    result = cv2.warpPerspective(
        image2,
        translation @ H,
        (xmax - xmin, ymax - ymin)
    )

    result[
        -ymin:h1 - ymin,
        -xmin:w1 - xmin
    ] = image1

    return result, matches, mask