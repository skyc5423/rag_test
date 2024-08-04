ai_descriptions = [
    {
        "name": "SAM",
        "function": "Segment anything in the input image. It can also get an input prompt like bbox, point, coarse mask.",
        "input_format": ["input_image: np.ndarray", "bbox: list=None", "mask: list=None", "points: list=None"],
        "output_format": ["segment_output: list"]
    },
    {
        "name": "Human Pose Estimation",
        "function": "Detects all humans in the image and identifies keypoints or joints of each detected human.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["keypoints: list", "bbox: list"]
    },
    {
        "name": "Object Detection",
        "function": "Identifies and locates objects within an image, providing bounding boxes and class labels.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["bbox: list", "class_labels: list", "confidence_scores: list"]
    },
    {
        "name": "Facial Recognition",
        "function": "Recognizes and verifies faces in an image, providing identity matches and confidence scores.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["identities: list", "confidence_scores: list"]
    },
    {
        "name": "Emotion Detection",
        "function": "Analyzes facial expressions in an image to determine the emotional state of detected faces.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["emotions: list", "confidence_scores: list"]
    },
    {
        "name": "Activity Recognition",
        "function": "Identifies and categorizes the activities or actions being performed in a video clip or image sequence.",
        "input_format": ["input_video: list of np.ndarray"],
        "output_format": ["activities: list", "confidence_scores: list"]
    },
    {
        "name": "Crowd Counting",
        "function": "Estimates the number of people present in an image or video frame.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["count: int"]
    },
    {
        "name": "Vehicle Detection and Tracking",
        "function": "Detects and tracks vehicles across frames in a video, providing bounding boxes and trajectories.",
        "input_format": ["input_video: list of np.ndarray"],
        "output_format": ["vehicle_tracks: list"]
    },
    {
        "name": "License Plate Recognition",
        "function": "Detects and recognizes license plates from images or video frames.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["license_plates: list"]
    },
    {
        "name": "Anomaly Detection",
        "function": "Identifies unusual or abnormal events or objects in an image or video.",
        "input_format": ["input_image: np.ndarray", "input_video: list of np.ndarray"],
        "output_format": ["anomalies: list", "confidence_scores: list"]
    },
    {
        "name": "Scene Text Recognition",
        "function": "Detects and recognizes text within natural scene images.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["text: str"]
    },
    {
        "name": "Semantic Segmentation",
        "function": "Assigns a class label to each pixel in the image.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["segmented_image: np.ndarray"]
    },
    {
        "name": "Instance Segmentation",
        "function": "Detects objects in an image and segments each instance of the object separately.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["instances: list of np.ndarray"]
    },
    {
        "name": "Optical Character Recognition (OCR)",
        "function": "Converts images of text into machine-encoded text.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["recognized_text: str"]
    },
    {
        "name": "3D Human Pose Estimation",
        "function": "Estimates 3D coordinates of human joints from a single image or a sequence of images.",
        "input_format": ["input_image: np.ndarray", "input_video: list of np.ndarray"],
        "output_format": ["3d_keypoints: list"]
    },
    {
        "name": "Hand Gesture Recognition",
        "function": "Identifies and classifies hand gestures in images or video frames.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["gestures: list"]
    },
    {
        "name": "Facial Landmark Detection",
        "function": "Detects facial landmarks (e.g., eyes, nose, mouth) in an image.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["landmarks: list"]
    },
    {
        "name": "Person Re-Identification",
        "function": "Identifies whether a person seen in one camera view is the same as a person seen in another view.",
        "input_format": ["input_image: np.ndarray", "gallery_images: list of np.ndarray"],
        "output_format": ["identity_matches: list", "confidence_scores: list"]
    },
    {
        "name": "Gaze Estimation",
        "function": "Estimates where a person is looking in an image.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["gaze_direction: list"]
    },
    {
        "name": "Pose Tracking",
        "function": "Tracks the pose of humans across multiple video frames.",
        "input_format": ["input_video: list of np.ndarray"],
        "output_format": ["pose_tracks: list"]
    },
    {
        "name": "Image Super-Resolution",
        "function": "Enhances the resolution of input images.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["super_res_image: np.ndarray"]
    },
    {
        "name": "Image Inpainting",
        "function": "Fills in missing or corrupted parts of an image.",
        "input_format": ["input_image: np.ndarray", "mask: np.ndarray"],
        "output_format": ["inpainted_image: np.ndarray"]
    },
    {
        "name": "Depth Estimation",
        "function": "Estimates the depth map of a scene from a single image.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["depth_map: np.ndarray"]
    },
    {
        "name": "Scene Classification",
        "function": "Classifies the overall scene depicted in an image into predefined categories.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["scene_label: str", "confidence_score: float"]
    },
    {
        "name": "Image Captioning",
        "function": "Generates descriptive captions for images.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["caption: str"]
    },
    {
        "name": "Video Summarization",
        "function": "Creates a concise summary of a video by selecting key frames or segments.",
        "input_format": ["input_video: list of np.ndarray"],
        "output_format": ["summary_video: list of np.ndarray"]
    },
    {
        "name": "Colorization",
        "function": "Adds color to grayscale images.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["colorized_image: np.ndarray"]
    },
    {
        "name": "Image Denoising",
        "function": "Removes noise from images.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["denoised_image: np.ndarray"]
    },
    {
        "name": "Salient Object Detection",
        "function": "Detects the most salient object in an image.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["salient_object: np.ndarray"]
    },
    {
        "name": "Medical Image Segmentation",
        "function": "Segments anatomical structures in medical images.",
        "input_format": ["input_image: np.ndarray"],
        "output_format": ["segmented_structures: list of np.ndarray"]
    }
]
