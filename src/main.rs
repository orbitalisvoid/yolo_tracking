use std::io::BufRead;

use opencv::{
    core::{MatTraitConst, MatTraitConstManual},
    dnn::{NetTrait, NetTraitConst},
    videoio::VideoCaptureTrait,
};

#[derive(Debug, Clone)]
struct Detection {
    class_id: i32,
    confidence: f32,
    box_: opencv::core::Rect,
}

struct ObjectDetection {
    net_: opencv::dnn::Net,
    classses_: Vec<String>,
    model_weights_: String,
    model_config_: String,
    classes_file_: String,
    conf_threshold_: f32,
    nms_threshold_: f32,
    load_classes: fn(&str) -> Vec<String>,
}

impl ObjectDetection {
    fn new(
        model_weights: &str,
        model_config: &str,
        classes_file: &str,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut net = opencv::dnn::read_net_from_darknet(model_config, model_weights).unwrap();

        if net.empty()? {
            return Err("Failed to load network".into());
        }

        net.set_preferable_backend(opencv::dnn::DNN_BACKEND_OPENCV)
            .unwrap();
        net.set_preferable_target(opencv::dnn::DNN_TARGET_CPU)
            .unwrap();

        let classes = Self::load_classes(classes_file);

        Ok(ObjectDetection {
            net_: net,
            classses_: classes,
            model_weights_: model_weights.to_string(),
            model_config_: model_config.to_string(),
            classes_file_: classes_file.to_string(),
            conf_threshold_: conf_threshold,
            nms_threshold_: nms_threshold,
            load_classes: Self::load_classes,
        })
    }

    fn load_classes(classes_file: &str) -> Vec<String> {
        let file = std::fs::File::open(classes_file).unwrap();
        let reader = std::io::BufReader::new(file);
        let classes = reader
            .lines()
            .map(|line| line.unwrap())
            .collect::<Vec<String>>();
        return classes;
    }

    fn detect(&mut self, frame: opencv::core::Mat) -> Vec<Detection> {
        let mut float_frame = opencv::core::Mat::default();
        frame
            .convert_to(&mut float_frame, opencv::core::CV_32F, 1.0 / 255.0, 0.0)
            .unwrap();

        let blob = opencv::dnn::blob_from_image(
            &float_frame, // Use float_frame instead of frame
            1.0,
            opencv::core::Size::new(416, 416),
            opencv::core::Scalar::new(0.0, 0.0, 0.0, 0.0),
            true,
            false,
            0,
        )
        .unwrap();

        self.net_
            .set_input(
                &blob,
                "",
                1.0,
                opencv::core::Scalar::new(0.0, 0.0, 0.0, 0.0),
            )
            .unwrap();

        let mut outputs: opencv::core::Vector<opencv::core::Mat> = opencv::core::Vector::new();

        let mut out_blob_names: opencv::core::Vector<String> = self.get_output_names().unwrap();

        self.net_
            .forward(&mut outputs, &mut out_blob_names)
            .unwrap();

        match self.process_detections(&frame, &outputs) {
            Ok(detections) => detections,
            Err(_) => vec![],
        }
    }

    fn process_detections(
        &self,
        image: &opencv::core::Mat,
        outputs: &opencv::core::Vector<opencv::core::Mat>,
    ) -> Result<Vec<Detection>, Box<dyn std::error::Error>> {
        let mut class_ids = Vec::new();
        let mut confidences = Vec::new();
        let mut boxes = Vec::new();

        let width = image.cols();
        let height = image.rows();

        for output in outputs.iter() {
            let data = output.data_typed::<f32>()?;
            let num_detections = output.rows();
            let num_attributes = output.cols();

            for i in 0..num_detections {
                let start_idx = i * num_attributes;
                let scores = &data[(start_idx + 5) as usize..(start_idx + num_attributes) as usize];

                let (max_idx, &max_conf) = scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();

                if max_conf > self.conf_threshold_ {
                    let center_x = data[0] * width as f32;
                    let center_y = data[1] * height as f32;
                    let box_width = data[2] * width as f32;
                    let box_height = data[3] * height as f32;

                    let left = (center_x - box_width / 2.0) as i32;
                    let top = (center_y - box_height / 2.0) as i32;

                    class_ids.push(max_idx as i32);
                    confidences.push(max_conf);
                    boxes.push(opencv::core::Rect::new(
                        left,
                        top,
                        box_width as i32,
                        box_height as i32,
                    ));
                }
            }
        }

        let mut indices = opencv::core::Vector::<i32>::new();
        opencv::dnn::nms_boxes(
            &opencv::core::Vector::from_iter(boxes.clone()),
            &opencv::core::Vector::from_iter(confidences.clone()),
            self.conf_threshold_,
            self.nms_threshold_,
            &mut indices,
            1.0,
            0,
        )?;

        let mut detections = Vec::new();
        for idx in indices.iter() {
            detections.push(Detection {
                class_id: class_ids[idx as usize],
                confidence: confidences[idx as usize],
                box_: boxes[idx as usize].clone(),
            });
        }

        Ok(detections)
    }

    fn get_output_names(&self) -> Result<opencv::core::Vector<String>, Box<dyn std::error::Error>> {
        let mut names = opencv::core::Vector::<String>::new();
        let out_layers = self.net_.get_unconnected_out_layers_names()?;
        for name in out_layers.iter() {
            names.push(&name.to_string());
        }
        Ok(names)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const YOLO_FOLDER_PATH: &str =
        "/Users/lazycodebaker/Documents/Code/Personal/Rust/tracking_opencv/yolo";

    let mut detector = ObjectDetection::new(
        std::path::Path::new(YOLO_FOLDER_PATH)
            .join("yolov4.weights")
            .to_str()
            .unwrap(),
        std::path::Path::new(YOLO_FOLDER_PATH)
            .join("yolov4.cfg")
            .to_str()
            .unwrap(),
        std::path::Path::new(YOLO_FOLDER_PATH)
            .join("coco.names")
            .to_str()
            .unwrap(),
        0.5,
        0.5,
    )
    .unwrap();

    let video_path = std::path::Path::new(YOLO_FOLDER_PATH).join("0.mp4");
    let video_file = video_path.to_str().unwrap();

    let mut cap =
        opencv::videoio::VideoCapture::from_file(video_file, opencv::videoio::CAP_ANY).unwrap();

    opencv::highgui::named_window("Object Tracking", opencv::highgui::WINDOW_NORMAL)?;

    loop {
        let mut frame = opencv::core::Mat::default();
        cap.read(&mut frame)?;

        let detections = detector.detect(frame);

        for detection in detections {
            let class_name = &detector.classses_[detection.class_id as usize];

            println!(
                "Class: {}, Confidence: {}",
                class_name, detection.confidence
            );
        }

        // Break the loop if the 'q' key is pressed
        if opencv::highgui::wait_key(1)? == 'q' as i32 {
            break;
        }
    }

    Ok(())
}
