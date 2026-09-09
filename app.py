import streamlit as st
import cv2
import numpy as np


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PhotoMatch",
    page_icon="",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

/* -------------------- PAGE -------------------- */

.stApp {
    background-color: #ffffff;
}

.block-container {
    max-width: 1150px;
    padding-top: 30px;
    padding-bottom: 60px;
}


/* -------------------- HIDE STREAMLIT UI -------------------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* Remove heading anchor icons */

a[href^="#"] {
    display: none !important;
}


/* -------------------- TEXT -------------------- */

h1, h2, h3, h4 {
    color: #292326 !important;
}

p {
    color: #292326;
}


/* -------------------- NAVBAR -------------------- */

.nav-title {
    color: #292326;
    font-size: 24px;
    font-weight: 700;
}

.nav-subtitle {
    color: #81747a;
    font-size: 14px;
    text-align: right;
}


/* -------------------- HERO -------------------- */

.badge {
    color: #d26890;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    text-align: center;
}

.hero-title {
    color: #292326;
    font-size: 50px;
    font-weight: 750;
    line-height: 1.15;
    text-align: center;
}

.hero-pink {
    color: #d86b91;
}

.hero-description {
    color: #81747a;
    font-size: 17px;
    line-height: 1.7;
    text-align: center;
    max-width: 720px;
    margin: auto;
}


/* -------------------- UPLOAD -------------------- */

.upload-title {
    color: #292326;
    font-size: 24px;
    font-weight: 700;
    text-align: center;
}

.upload-description {
    color: #81747a;
    font-size: 14px;
    text-align: center;
}


/* File uploader */

[data-testid="stFileUploader"] {
    background-color: #fff7fa;
    border: 2px dashed #e8b6c8;
    border-radius: 18px;
    padding: 12px;
}


/* Browse button */

[data-testid="stFileUploader"] button {
    background-color: #d86b91 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
}


/* -------------------- FILE NAMES -------------------- */

.file-name {
    color: #292326 !important;
    font-size: 15px;
    font-weight: 600;
    padding: 6px 0;
}


/* -------------------- BUTTON -------------------- */

.stButton > button {
    width: 100%;
    background-color: #d86b91;
    color: white;
    border: none;
    border-radius: 11px;
    padding: 13px;
    font-size: 16px;
    font-weight: 650;
}

.stButton > button:hover {
    background-color: #c75c83;
    color: white;
}


/* -------------------- SECTION TITLES -------------------- */

.section-title {
    color: #292326;
    font-size: 30px;
    font-weight: 700;
    text-align: center;
}

.section-description {
    color: #81747a;
    font-size: 15px;
    text-align: center;
}


/* -------------------- FEATURE CARDS -------------------- */

.feature-card {
    background-color: #ffffff;
    border: 1px solid #f0dce4;
    border-radius: 18px;
    padding: 25px;
    min-height: 205px;
    box-shadow: 0 8px 25px rgba(216, 107, 145, 0.07);
}

.feature-number {
    color: #d86b91;
    font-size: 14px;
    font-weight: 700;
}

.feature-title {
    color: #292326;
    font-size: 19px;
    font-weight: 700;
}

.feature-text {
    color: #81747a;
    font-size: 14px;
    line-height: 1.6;
}


/* -------------------- OUTPUT -------------------- */

.output-box {
    background-color: #fff7fa;
    border: 1px solid #f0dce4;
    border-radius: 18px;
    padding: 25px;
}


/* -------------------- FOOTER -------------------- */

.footer-text {
    color: #9b8c92;
    font-size: 13px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def load_image(uploaded_file):
    """
    Convert Streamlit uploaded file into OpenCV image.
    """

    # Reset file pointer to the beginning
    uploaded_file.seek(0)

    file_bytes = np.frombuffer(
        uploaded_file.read(),
        dtype=np.uint8
    )

    if file_bytes.size == 0:
        raise ValueError(f"Could not read file: {uploaded_file.name}")

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"OpenCV could not decode: {uploaded_file.name}")

    return image


def resize_image(image, max_width=1200):
    """
    Resize image while keeping aspect ratio.
    """

    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def stitch_two_images(base, new_image):
    """
    Stitch new_image onto base using:

    SIFT
    ↓
    Feature matching
    ↓
    Lowe's ratio test
    ↓
    RANSAC
    ↓
    Homography
    ↓
    Perspective warping
    """

    # ---------------------------------------------
    # Resize
    # ---------------------------------------------

    base = resize_image(base)
    new_image = resize_image(new_image)


    # ---------------------------------------------
    # Convert to grayscale
    # ---------------------------------------------

    gray_base = cv2.cvtColor(
        base,
        cv2.COLOR_BGR2GRAY
    )

    gray_new = cv2.cvtColor(
        new_image,
        cv2.COLOR_BGR2GRAY
    )


    # ---------------------------------------------
    # SIFT FEATURE DETECTION
    # ---------------------------------------------

    sift = cv2.SIFT_create()

    keypoints_base, descriptors_base = sift.detectAndCompute(
        gray_base,
        None
    )

    keypoints_new, descriptors_new = sift.detectAndCompute(
        gray_new,
        None
    )


    if descriptors_base is None or descriptors_new is None:
        raise ValueError(
            "Could not detect enough features in the images."
        )


    # ---------------------------------------------
    # FEATURE MATCHING
    # ---------------------------------------------

    matcher = cv2.BFMatcher()

    matches = matcher.knnMatch(
        descriptors_new,
        descriptors_base,
        k=2
    )


    # ---------------------------------------------
    # LOWE'S RATIO TEST
    # ---------------------------------------------

    good_matches = []

    for pair in matches:

        if len(pair) == 2:

            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)


    if len(good_matches) < 10:

        raise ValueError(
            f"Not enough matching features. "
            f"Only {len(good_matches)} good matches found."
        )


    # ---------------------------------------------
    # GET MATCHING POINTS
    # ---------------------------------------------

    src_points = np.float32([
        keypoints_new[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)


    dst_points = np.float32([
        keypoints_base[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)


    # ---------------------------------------------
    # HOMOGRAPHY + RANSAC
    # ---------------------------------------------

    H, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        5.0
    )


    if H is None:
        raise ValueError(
            "Homography could not be calculated."
        )


    # ---------------------------------------------
    # FIND IMAGE CORNERS
    # ---------------------------------------------

    h1, w1 = base.shape[:2]
    h2, w2 = new_image.shape[:2]

    corners_new = np.float32([
        [0, 0],
        [0, h2],
        [w2, h2],
        [w2, 0]
    ]).reshape(-1, 1, 2)

    transformed_corners = cv2.perspectiveTransform(
        corners_new,
        H
    )


    corners_base = np.float32([
        [0, 0],
        [0, h1],
        [w1, h1],
        [w1, 0]
    ]).reshape(-1, 1, 2)


    all_corners = np.concatenate(
        (corners_base, transformed_corners),
        axis=0
    )


    [xmin, ymin] = np.int32(
        all_corners.min(axis=0).ravel() - 10
    )

    [xmax, ymax] = np.int32(
        all_corners.max(axis=0).ravel() + 10
    )


    # ---------------------------------------------
    # TRANSLATION
    # ---------------------------------------------

    translation = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1]
    ])


    # ---------------------------------------------
    # WARP NEW IMAGE
    # ---------------------------------------------

    result_width = xmax - xmin
    result_height = ymax - ymin

    panorama = cv2.warpPerspective(
        new_image,
        translation @ H,
        (result_width, result_height)
    )


    # ---------------------------------------------
    # PLACE BASE IMAGE
    # ---------------------------------------------

    x_offset = -xmin
    y_offset = -ymin

    panorama[
        y_offset:y_offset + h1,
        x_offset:x_offset + w1
    ] = np.maximum(
        panorama[
            y_offset:y_offset + h1,
            x_offset:x_offset + w1
        ],
        base
    )


    # ---------------------------------------------
    # CROP BLACK AREA
    # ---------------------------------------------

    gray = cv2.cvtColor(
        panorama,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        1,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if contours:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            largest
        )

        panorama = panorama[
            y:y+h,
            x:x+w
        ]


    return panorama, len(good_matches)


# ============================================================
# NAVBAR
# ============================================================

left, right = st.columns([2, 1])

with left:

    st.markdown(
        '<div class="nav-title">PhotoMatch</div>',
        unsafe_allow_html=True
    )

with right:

    st.markdown(
        '<div class="nav-subtitle">'
        'Computer Vision · Feature Matching'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# HERO
# ============================================================

st.write("")

st.markdown(
    '<div class="badge">'
    'AUTOMATIC IMAGE ALIGNMENT'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.markdown(
    '<div class="hero-title">'
    'Turn Multiple Photos Into<br>'
    '<span class="hero-pink">One Seamless Panorama</span>'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.markdown(
    '<div class="hero-description">'
    'Upload overlapping photographs and let computer vision '
    'detect features, match corresponding points, align the '
    'images and create one panoramic collage.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD
# ============================================================

st.write("")
st.write("")
st.write("")

st.markdown(
    '<div class="upload-title">'
    'Create Your Photo Collage'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-description">'
    'Upload 2–3 overlapping photographs of the same scene.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


uploaded_files = st.file_uploader(
    "Select your photographs",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)


# ============================================================
# DISPLAY FILE NAMES
# ============================================================

if uploaded_files:

    st.write("")

    st.markdown(
        '<p style="color:#292326; font-weight:700;">'
        'Selected Images'
        '</p>',
        unsafe_allow_html=True
    )

    for i, file in enumerate(uploaded_files):

        st.markdown(
            f'<div class="file-name">'
            f'{i + 1}. {file.name}'
            f'</div>',
            unsafe_allow_html=True
        )


# ============================================================
# PREVIEW IMAGES
# ============================================================

if uploaded_files:

    st.write("")
    st.markdown(
        '<p style="color:#292326; font-weight:700;">'
        'Image Preview'
        '</p>',
        unsafe_allow_html=True
    )

    preview_columns = st.columns(
        min(len(uploaded_files), 3)
    )

    for i, file in enumerate(uploaded_files):

        image = load_image(file)

        if image is not None:

            with preview_columns[i % 3]:

                rgb_image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                # Compatible with older Streamlit versions
                st.image(
                    rgb_image,
                    caption=file.name
                )


# ============================================================
# CREATE PANORAMA BUTTON
# ============================================================

st.write("")
st.write("")

if len(uploaded_files) >= 2:

    if st.button(
        "Create Panorama"
    ):

        if len(uploaded_files) > 3:

            st.warning(
                "Please select only 2 or 3 images."
            )

        else:

            with st.spinner(
                "Detecting features and creating panorama..."
            ):

                try:

                    # -----------------------------------------
                    # LOAD IMAGES
                    # -----------------------------------------

                    images = []

                    for file in uploaded_files:

                        image = load_image(file)

                        if image is None:
                            raise ValueError(
                                f"Could not read {file.name}"
                            )

                        images.append(image)


                    # -----------------------------------------
                    # START WITH FIRST IMAGE
                    # -----------------------------------------

                    panorama = images[0]

                    total_matches = []


                    # -----------------------------------------
                    # STITCH EACH NEXT IMAGE
                    # -----------------------------------------

                    for i in range(1, len(images)):

                        panorama, matches_count = stitch_two_images(
                            panorama,
                            images[i]
                        )

                        total_matches.append(
                            matches_count
                        )


                    # -----------------------------------------
                    # SAVE OUTPUT
                    # -----------------------------------------

                    output_path = "output/panorama.jpg"

                    cv2.imwrite(
                        output_path,
                        panorama
                    )


                    # -----------------------------------------
                    # SHOW RESULT
                    # -----------------------------------------

                    st.success(
                        "Panorama created successfully!"
                    )


                    st.write("")

                    st.markdown(
                        '<div class="section-title">'
                        'Your Final Panorama'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.write("")


                    panorama_rgb = cv2.cvtColor(
                        panorama,
                        cv2.COLOR_BGR2RGB
                    )


                    st.image(
                        panorama_rgb,
                        caption="Final Photo Collage"
                    )


                    # -----------------------------------------
                    # MATCH INFORMATION
                    # -----------------------------------------

                    st.write("")

                    info1, info2, info3 = st.columns(3)

                    with info1:

                        st.metric(
                            "Images",
                            len(images)
                        )

                    with info2:

                        st.metric(
                            "Feature Matches",
                            sum(total_matches)
                        )

                    with info3:

                        st.metric(
                            "Method",
                            "SIFT + RANSAC"
                        )


                    # -----------------------------------------
                    # DOWNLOAD
                    # -----------------------------------------

                    with open(
                        output_path,
                        "rb"
                    ) as file:

                        st.download_button(
                            "Download Panorama",
                            data=file,
                            file_name="photo_match_panorama.jpg",
                            mime="image/jpeg"
                        )


                except Exception as e:

                    st.error(
                        f"Panorama creation failed: {str(e)}"
                    )

else:

    st.info(
        "Upload at least 2 overlapping images to create a panorama."
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.write("")
st.write("")
st.write("")

st.markdown(
    '<div class="section-title">'
    'How PhotoMatch Works'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'A computer vision pipeline automatically aligns '
    'overlapping photographs.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")
st.write("")


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        <div class="feature-card">

        <div class="feature-number">
        01
        </div>

        <br>

        <div class="feature-title">
        Detect Features
        </div>

        <br>

        <div class="feature-text">
        SIFT detects distinctive points such as
        corners, edges and patterns in each image.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="feature-card">

        <div class="feature-number">
        02
        </div>

        <br>

        <div class="feature-title">
        Match & Align
        </div>

        <br>

        <div class="feature-text">
        Feature matching finds corresponding points.
        RANSAC calculates a reliable homography.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="feature-card">

        <div class="feature-number">
        03
        </div>

        <br>

        <div class="feature-title">
        Create Panorama
        </div>

        <br>

        <div class="feature-text">
        The images are warped, aligned and combined
        into one larger panoramic photograph.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.write("")
st.write("")
st.divider()

st.markdown(
    '<div class="footer-text">'
    'Built with Python · OpenCV · SIFT · RANSAC · Homography'
    '</div>',
    unsafe_allow_html=True
)