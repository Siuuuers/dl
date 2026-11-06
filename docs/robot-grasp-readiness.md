# From cassette corners to a robot grasp

**Purpose confirmed by the user on 2026-11-07 (Hong Kong): robotic positioning/grasping.** Continue the controlled two-corner model comparison while establishing the geometry and robot contract below. No physical tolerance, camera mounting arrangement or robot calibration has yet been supplied. This document records the next implementation choices; it does not report a functioning robot interface or a grasp-accuracy result.

## What the existing project actually provides

Source inspected: `Siuuuers/dl` at `c4f82c039fee3087e8a3fb36fe6c9f735acdb5dc`.

| Component | Observed implementation | Consequence |
| --- | --- | --- |
| [D405Capture](../scripts/view_d405_yolo.py) | Enables only the BGR8 color stream. Records device/profile identity, frame number, device timestamp and host monotonic receive time. | A D405 name does not establish that depth was captured. |
| [save_snapshot](../scripts/view_d405_yolo.py) | Saves original-color PNG, overlay PNG and JSON with 2D detections in original-image pixels. | These snapshots do not contain a measured 3D corner or robot pose. |
| [RawVideoRecorder](../scripts/d405_recording.py) | Saves color MP4, timing CSV and metadata; the raw mode means unannotated color. | It is not a lossless depth recording. Video timing is separate from playback frame rate. |
| Camera/robot geometry | The inspected camera paths do not obtain intrinsics, depth, stream extrinsics, camera-to-robot transforms or TCP calibration. No robot driver or metric-depth/calibration artifact was found in the inspected tracked tree. | Millimetres, orientation and robot-base coordinates remain unestablished. |
| [Model quality](model-quality.md) | Frozen two-corner development evaluation and a controlled training comparison. | Continue measuring pair availability and pixel error; neither currently certifies grasp performance. |

## Geometry that the model cannot supply by itself

A calibrated color pixel defines a camera ray. Metric depth or a justified geometric constraint is needed to locate a point along that ray. RealSense associates intrinsics with each stream profile, supports deprojection with depth, and uses camera axes X right, Y down, Z forward in metres. Depth Z is the optical-axis component, not radial range. [RealSense projection](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/) · [Projection and occlusion](https://dev.realsenseai.com/docs/projection-texture-mapping-and-occlusion-with-intel-realsense-depth-cameras/)

Two image corners alone do not determine an arbitrary six-degree-of-freedom grasp pose. Even two correctly measured 3D corners determine a midpoint and an edge direction while leaving rotation about that edge undetermined. This follows geometrically: rotating an object about the line through those two points leaves both points unchanged. A measured face normal, known fixture orientation or other supported geometry must resolve the remaining orientation.

For a distortion-corrected pinhole image, `X = (u - cx) * Z / fx`. At fixed depth and calibration, horizontal sensitivity is approximately `delta_X = (Z / fx) * delta_u`. This is a local sensitivity calculation, not total system error: depth, calibration, tool offsets and motion also contribute. The camera model and coordinate transformations are described in [OpenCV's calibration reference](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html). No constant pixels-to-millimetres factor can be assigned from the current PNGs alone.

The front-edge midpoint is an observable feature. The desired tool contact point, jaw orientation, approach direction and insertion/extraction offset must be defined separately. Image-X endpoint ordering also does not establish persistent physical left/right identities across arbitrary object rotations.

## Reuse useful upstream work without changing this task

The [grasp-vision README](https://github.com/an94-doll/embedding_vision/blob/9a6bf2432582af517d789c94eab2ac27e03a304b/README.md) and [first_box.endpoint_geometry](https://github.com/an94-doll/embedding_vision/blob/9a6bf2432582af517d789c94eab2ac27e03a304b/scripts/first_box.py) provide relevant patterns: profile-matched intrinsics, explicit rays-only output, optional intersection with a measured plane, and separate geometry/selection status. [detect_d405.py](https://github.com/an94-doll/embedding_vision/blob/9a6bf2432582af517d789c94eab2ac27e03a304b/detect_d405.py) clears output on stale frames. Its default plane is absent, and it does not supply a robot driver.

These are future reuse candidates. Do not port its two-model basket stack or replace baomai's task merely to obtain these helpers. Its nominal 30 mm width, 7 mm layer spacing and mentioned historical 2 mm requirement have not been established for this project.

That branch's 477-image source includes the original 157 images, and its recorded split is by random frames. Its 320 additional images require target-semantic mapping, hash/near-duplicate checks and capture-group separation before any reuse. Its split must not be assumed independent of our data, and its reported 81.09% two-point pixel result belongs to a different model; it does not establish this model's performance or robot grasp accuracy.

## Choose the smallest supported geometry route

| Actual setup | Suitable next implementation | Required evidence |
| --- | --- | --- |
| Target corners lie on a measured, stable plane | Undistort each corner and intersect its ray with that plane | Plane frame, normal, offset, calibration identity and operating envelope; verify that stack height/tilt does not violate the plane assumption |
| Target depth or tilt varies | Capture paired color and metric depth, then recover supported 3D points or a face plane | Matched profiles, depth scale, stream transform/alignment, valid timestamps and support from the intended surface |
| Camera fixed in the workcell | Calibrate camera to robot base | Measured transform and independent checks throughout the working volume |
| Camera mounted on the moving wrist | Calibrate camera to flange and combine it with robot pose at image capture | Hand-eye transform, timestamped robot pose and verified time alignment |

The mounting rows combine with either geometry row. OpenCV describes both fixed-camera and wrist-camera calibration, including transform direction; name frames explicitly rather than relying on an unlabeled matrix. [Hand-eye calibration](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)

For a wrist camera, using `T_A_from_B` to mean “map B coordinates into A”:

```text
T_base_from_camera(t) = T_base_from_flange(t) * T_flange_from_camera
T_base_from_tcp_goal = T_base_from_camera(t) * T_camera_from_object * T_object_from_tcp_goal
```

Use the fixed `T_base_from_camera` in a fixed-camera installation. Record TCP-to-flange calibration too, because the controller may expect a flange pose rather than a TCP pose.

Depth at an object boundary may belong to the rack, background or an occluder. Aligned frames still require valid surface correspondence; invalid Z16 depth is zero. Preserve invalid values and reject an unsupported corner/plane instead of filling the gap with a plausible grasp coordinate. [RealSense depth and alignment](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/) · [Occlusion limitations](https://dev.realsenseai.com/docs/projection-texture-mapping-and-occlusion-with-intel-realsense-depth-cameras/)

## Capture and acceptance contract for the next stage

A reproducible geometry sample should retain original color, paired lossless metric-depth data if used, frame/profile identity, intrinsics/distortion, depth scale and stream transform. Record calibration hashes, units, transform directions, robot/tool identity and timestamped robot pose when required. Preserve any crop/resize mapping. Keep raw and filtered depth distinguishable.

A proposed perception output should separate observation validity, geometry validity and target-selection validity. Include capture time and clock domain, frame/sequence, model hash, corner availability, selected instance, coordinate frame and calibration identity. Host monotonic timestamps cannot be compared directly across machines. A stale observation, missing corner, ambiguous target or unsupported geometry must produce an unavailable grasp proposal with a reason, not a reused coordinate presented as current. This is a future interface contract, not an implemented controller.

Before hardware acceptance, specify:

| Quantity | Measurement and denominator |
| --- | --- |
| Position | Error of the intended contact/TCP point in the robot frame, in mm; report axes and error distribution against independently measured references |
| Orientation | Error in the controlled rotational degrees of freedom, with legitimate gripper symmetries defined beforehand |
| Availability | Useful proposals per eligible grasp opportunity, including misses and abstentions; report incorrect valid proposals separately |
| Physical success | Correct target acquired and retained through the agreed action, with all attempted grasps and rejected opportunities reported |
| Timing | Capture-to-available-pose latency and observation age when used by the robot |
| Generalization | Untouched capture sessions and scene configurations, separate from training, tuning and calibration-fit observations |

The robot/gripper geometry and task owner must set the limits; the existing 2/5/10 px diagnostics remain illustrative. Check accuracy at independent physical locations, not only calibration reprojection residuals. Count abstention separately so a system cannot appear successful merely by attempting the easiest cases. Preserve failures by cause: wrong target, unavailable geometry, positioning, orientation, contact or retention.

## Immediate continuation

Continue checkpoint provenance and the [matched training experiment](prepare-pose-training.md); the robotic purpose does not require changing the current two-point labels. Collect the missing installation facts in parallel, then implement one geometry route and validate it offline with recorded samples before connecting it to motion.

The three questions that change the design are:

1. Is the camera fixed beside the robot or mounted on its wrist, and is a calibration already available?
2. Are cassette corners constrained to known planes, or can depth/tilt vary? Can paired depth and robot-pose recordings be obtained?
3. Which robot/gripper and contact point are intended, and what position, angle, success-rate and timing limits are required?

Unknown answers do not block the controlled model comparison. They do block an honest millimetre/robot-grasp acceptance claim. No camera acquisition, calibration experiment or robot motion was executed for this document.
