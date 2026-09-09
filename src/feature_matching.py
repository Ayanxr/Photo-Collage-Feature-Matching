import cv2


def detect_features(image):
    """
    Detect SIFT features and extract descriptors.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    return keypoints, descriptors


def match_features(descriptors1, descriptors2):
    """
    Match features between two images using
    Brute Force Matcher and Lowe's ratio test.
    """

    matcher = cv2.BFMatcher()

    matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for m, n in matches:

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches


def draw_matches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    matches
):
    """
    Draw matched feature points between two images.
    """

    result = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return result