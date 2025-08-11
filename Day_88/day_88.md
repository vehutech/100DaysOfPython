# AI Mastery Course - Day 88: Computer Vision Advanced Topics

## Learning Objective
By the end of this lesson, you will master advanced computer vision techniques including object detection, facial recognition, image style transfer, and video analysis. You'll understand how these technologies work behind the scenes and implement them using Python and Django, preparing you to build sophisticated visual AI applications.

---

## Imagine That...

Imagine you're the head chef of the world's most advanced restaurant, where every dish is crafted not just with ingredients, but with the power of sight itself. Your culinary team doesn't just taste and smell - they can instantly identify every ingredient in a competitor's dish just by looking at it, recognize regular customers the moment they walk through the door, transform the visual style of any plate to match different cultural cuisines, and even track the movement of ingredients as they're being prepared. This is the realm of advanced computer vision - where machines learn to see, understand, and interpret the visual world with superhuman precision.

Today, we'll master the art of teaching machines to see like master chefs see - with depth, recognition, creativity, and temporal awareness.

---

## 1. Object Detection and Segmentation

Just as a master chef can instantly identify and separate each ingredient in a complex dish, object detection allows our AI to locate and classify multiple objects within a single image, while segmentation precisely outlines each object's boundaries.

### The Art of Visual Ingredient Recognition

```python
# models.py - Django model for storing detection results
from django.db import models
import json

class DetectionResult(models.Model):
    image = models.ImageField(upload_to='detections/')
    detected_objects = models.JSONField(default=list)
    confidence_scores = models.JSONField(default=list)
    bounding_boxes = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def add_detection(self, class_name, confidence, bbox):
        """Add a new detection result"""
        self.detected_objects.append(class_name)
        self.confidence_scores.append(float(confidence))
        self.bounding_boxes.append(bbox)
    
    def get_high_confidence_detections(self, threshold=0.7):
        """Return only detections above confidence threshold"""
        return [
            {
                'object': obj, 
                'confidence': conf, 
                'bbox': bbox
            }
            for obj, conf, bbox in zip(
                self.detected_objects, 
                self.confidence_scores, 
                self.bounding_boxes
            )
            if conf >= threshold
        ]

# views.py - Object detection view
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
from ultralytics import YOLO
import json

class ObjectDetectionView:
    def __init__(self):
        # Load pre-trained YOLO model (like having a trained chef's eye)
        self.model = YOLO('yolov8n.pt')
        
    @csrf_exempt
    def detect_objects(self, request):
        if request.method == 'POST' and request.FILES.get('image'):
            image_file = request.FILES['image']
            
            # Convert uploaded image to OpenCV format
            image_array = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            # Perform detection (like a chef analyzing a dish)
            results = self.model(image)
            
            # Create detection result instance
            detection_result = DetectionResult.objects.create(
                image=image_file
            )
            
            # Process each detection
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Store detection
                        detection_result.add_detection(
                            class_name, 
                            confidence, 
                            [float(x1), float(y1), float(x2), float(y2)]
                        )
            
            detection_result.save()
            
            return JsonResponse({
                'success': True,
                'detection_id': detection_result.id,
                'detections': detection_result.get_high_confidence_detections()
            })
        
        return render(request, 'detection/upload.html')

# Advanced segmentation for precise ingredient boundaries
class ImageSegmentation:
    def __init__(self):
        self.model = YOLO('yolov8n-seg.pt')  # Segmentation model
    
    def segment_image(self, image_path):
        """Perform instance segmentation"""
        results = self.model(image_path)
        
        segmentation_data = []
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                
                for i, mask in enumerate(masks):
                    class_id = int(boxes.cls[i].cpu().numpy())
                    confidence = boxes.conf[i].cpu().numpy()
                    class_name = self.model.names[class_id]
                    
                    segmentation_data.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'mask': mask.tolist()  # Binary mask for precise boundaries
                    })
        
        return segmentation_data
```

**Syntax Explanation:**
- `models.JSONField()`: Stores JSON data directly in Django models, perfect for flexible detection results
- `@csrf_exempt`: Disables CSRF protection for API endpoints (use carefully in production)
- `np.frombuffer()`: Converts uploaded file bytes to NumPy array for OpenCV processing
- `box.xyxy[0]`: Extracts bounding box coordinates in x1,y1,x2,y2 format
- `masks.data.cpu().numpy()`: Moves tensor data from GPU to CPU and converts to NumPy array

---

## 2. Facial Recognition Systems

Like a master chef who never forgets a regular customer's preferences, facial recognition systems can identify and authenticate individuals with remarkable accuracy.

### The Personal Touch Recognition System

```python
# models.py - Customer recognition system
import face_recognition
import pickle
import os
from django.conf import settings

class CustomerProfile(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    photo = models.ImageField(upload_to='customer_photos/')
    face_encoding = models.BinaryField()  # Store face encoding
    visit_count = models.IntegerField(default=0)
    last_visit = models.DateTimeField(auto_now=True)
    preferences = models.JSONField(default=dict)
    
    def save(self, *args, **kwargs):
        if self.photo and not self.face_encoding:
            self.generate_face_encoding()
        super().save(*args, **kwargs)
    
    def generate_face_encoding(self):
        """Generate and store face encoding from photo"""
        image_path = os.path.join(settings.MEDIA_ROOT, str(self.photo))
        image = face_recognition.load_image_file(image_path)
        
        # Get face encodings (128-dimensional face fingerprint)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            # Store the first face encoding found
            self.face_encoding = pickle.dumps(encodings[0])
    
    def get_face_encoding(self):
        """Retrieve face encoding from binary storage"""
        if self.face_encoding:
            return pickle.loads(self.face_encoding)
        return None

class FacialRecognitionService:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load all known customer faces"""
        customers = CustomerProfile.objects.all()
        self.known_encodings = []
        self.known_names = []
        
        for customer in customers:
            encoding = customer.get_face_encoding()
            if encoding is not None:
                self.known_encodings.append(encoding)
                self.known_names.append(customer.name)
    
    def recognize_face(self, image_path):
        """Recognize faces in uploaded image"""
        # Load and process the image
        unknown_image = face_recognition.load_image_file(image_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        unknown_locations = face_recognition.face_locations(unknown_image)
        
        recognized_faces = []
        
        for face_encoding, face_location in zip(unknown_encodings, unknown_locations):
            # Compare against known faces
            matches = face_recognition.compare_faces(
                self.known_encodings, 
                face_encoding,
                tolerance=0.6  # Adjust for strictness
            )
            
            # Calculate face distances (lower = better match)
            distances = face_recognition.face_distance(
                self.known_encodings, 
                face_encoding
            )
            
            best_match_index = np.argmin(distances)
            
            if matches[best_match_index] and distances[best_match_index] < 0.6:
                name = self.known_names[best_match_index]
                confidence = 1 - distances[best_match_index]
                
                # Update customer visit
                customer = CustomerProfile.objects.get(name=name)
                customer.visit_count += 1
                customer.save()
                
                recognized_faces.append({
                    'name': name,
                    'confidence': confidence,
                    'location': face_location,
                    'customer_id': customer.id
                })
            else:
                recognized_faces.append({
                    'name': 'Unknown',
                    'confidence': 0.0,
                    'location': face_location,
                    'customer_id': None
                })
        
        return recognized_faces

# views.py
from django.views.generic import View

class CustomerRecognitionView(View):
    def __init__(self):
        super().__init__()
        self.recognition_service = FacialRecognitionService()
    
    def post(self, request):
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            
            # Save temporary image
            temp_path = f'/tmp/{image_file.name}'
            with open(temp_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)
            
            # Recognize faces
            results = self.recognition_service.recognize_face(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return JsonResponse({
                'recognized_faces': results,
                'total_faces': len(results)
            })
        
        return JsonResponse({'error': 'No image provided'}, status=400)
```

**Syntax Explanation:**
- `models.BinaryField()`: Stores binary data (pickled face encodings) in database
- `pickle.dumps()/pickle.loads()`: Serializes/deserializes Python objects to/from binary format
- `face_recognition.face_encodings()`: Generates 128-dimensional numerical representations of faces
- `np.argmin()`: Finds index of minimum value (best face match)
- `tolerance=0.6`: Recognition threshold - lower values = stricter matching

---

## 3. Image Style Transfer

Just as a chef can present the same dish in French, Italian, or Asian style, neural style transfer allows us to reimagine images in different artistic styles while preserving their content.

### The Artistic Transformation Workshop

```python
# models.py - Style transfer system
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
import torch.nn as nn
import torch.nn.functional as F

class ArtisticStyle(models.Model):
    name = models.CharField(max_length=100)
    style_image = models.ImageField(upload_to='styles/')
    description = models.TextField()
    popularity_score = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

class StyleTransferResult(models.Model):
    original_image = models.ImageField(upload_to='originals/')
    style = models.ForeignKey(ArtisticStyle, on_delete=models.CASCADE)
    stylized_image = models.ImageField(upload_to='stylized/')
    processing_time = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

# Neural Style Transfer Implementation
class StyleTransferNet(nn.Module):
    def __init__(self):
        super(StyleTransferNet, self).__init__()
        # Use VGG19 as feature extractor (pre-trained chef's knowledge)
        vgg = vgg19(pretrained=True).features
        
        # Extract specific layers for style and content
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.content_layers = ['conv4_2']
        
        self.vgg = nn.Sequential()
        self.content_losses = []
        self.style_losses = []
        
        # Build the network
        conv_num = 1
        for layer in vgg:
            if isinstance(layer, nn.Conv2d):
                name = f'conv{conv_num}_1'
                self.vgg.add_module(name, layer)
                
                if name in self.content_layers:
                    # Add content loss
                    target = self.vgg(content_img).detach()
                    content_loss = ContentLoss(target)
                    self.vgg.add_module(f'content_loss_{conv_num}', content_loss)
                    self.content_losses.append(content_loss)
                
                if name in self.style_layers:
                    # Add style loss
                    target_feature = self.vgg(style_img).detach()
                    style_loss = StyleLoss(target_feature)
                    self.vgg.add_module(f'style_loss_{conv_num}', style_loss)
                    self.style_losses.append(style_loss)
                
                conv_num += 1
            
            elif isinstance(layer, nn.ReLU):
                name = f'relu{conv_num}'
                # Replace in-place ReLU with out-of-place
                self.vgg.add_module(name, nn.ReLU(inplace=False))
            
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool{conv_num}'
                self.vgg.add_module(name, layer)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)  # Initialize to zero
    
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)
    
    def gram_matrix(self, input):
        """Calculate Gram matrix for style representation"""
        batch_size, channels, height, width = input.size()
        features = input.view(batch_size * channels, height * width)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * channels * height * width)
    
    def forward(self, input):
        gram = self.gram_matrix(input)
        self.loss = F.mse_loss(gram, self.target)
        return input

# Django service for style transfer
class StyleTransferService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
    
    def transfer_style(self, content_image_path, style_image_path, steps=300):
        """Apply neural style transfer"""
        import time
        start_time = time.time()
        
        # Load and preprocess images
        content_img = self.load_image(content_image_path)
        style_img = self.load_image(style_image_path)
        
        # Initialize the input image (start with content image)
        input_img = content_img.clone()
        input_img.requires_grad_(True)
        
        # Create the model
        model = StyleTransferNet()
        model.to(self.device)
        
        # Optimizer
        optimizer = torch.optim.LBFGS([input_img])
        
        run = [0]
        while run[0] <= steps:
            def closure():
                # Clamp input image values
                input_img.data.clamp_(0, 1)
                
                optimizer.zero_grad()
                model(input_img)
                
                style_score = 0
                content_score = 0
                
                for sl in model.style_losses:
                    style_score += sl.loss
                for cl in model.content_losses:
                    content_score += cl.loss
                
                # Weight the losses (favor content vs style)
                total_loss = content_score + style_score * 1000000
                total_loss.backward()
                
                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"Step {run[0]}: Style Loss: {style_score.item():.4f}, "
                          f"Content Loss: {content_score.item():.4f}")
                
                return total_loss
            
            optimizer.step(closure)
        
        # Final clamp
        input_img.data.clamp_(0, 1)
        
        processing_time = time.time() - start_time
        return input_img, processing_time
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        return image.to(self.device)
    
    def save_result(self, tensor, path):
        """Convert tensor back to image and save"""
        unloader = transforms.ToPILImage()
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        image.save(path)
        return path

# views.py
class StyleTransferView(View):
    def __init__(self):
        super().__init__()
        self.transfer_service = StyleTransferService()
    
    def post(self, request):
        content_image = request.FILES.get('content_image')
        style_id = request.POST.get('style_id')
        
        if not content_image or not style_id:
            return JsonResponse({'error': 'Missing image or style'}, status=400)
        
        try:
            style = ArtisticStyle.objects.get(id=style_id)
            
            # Save content image temporarily
            content_path = f'/tmp/content_{content_image.name}'
            with open(content_path, 'wb+') as destination:
                for chunk in content_image.chunks():
                    destination.write(chunk)
            
            # Perform style transfer
            result_tensor, processing_time = self.transfer_service.transfer_style(
                content_path, 
                style.style_image.path
            )
            
            # Save result
            result_path = f'/tmp/stylized_{content_image.name}'
            self.transfer_service.save_result(result_tensor, result_path)
            
            # Create database record
            with open(result_path, 'rb') as f:
                result = StyleTransferResult.objects.create(
                    style=style,
                    processing_time=processing_time
                )
                result.original_image.save(content_image.name, content_image)
                result.stylized_image.save(f'stylized_{content_image.name}', 
                                         ContentFile(f.read()))
            
            # Cleanup
            os.unlink(content_path)
            os.unlink(result_path)
            
            return JsonResponse({
                'success': True,
                'result_id': result.id,
                'stylized_image_url': result.stylized_image.url,
                'processing_time': processing_time
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
```

**Syntax Explanation:**
- `nn.Module`: Base class for all neural network modules in PyTorch
- `tensor.detach()`: Removes gradient computation to save memory
- `F.mse_loss()`: Mean Squared Error loss function
- `torch.mm()`: Matrix multiplication for Gram matrix computation
- `requires_grad_(True)`: Enables gradient computation for tensor optimization
- `optimizer.zero_grad()`: Clears gradients from previous iteration

---

## 4. Video Analysis and Tracking

Like a chef monitoring multiple dishes simultaneously, video analysis allows us to track objects, people, and activities across time, understanding motion patterns and temporal relationships.

### The Motion Choreography System

```python
# models.py - Video analysis system
class VideoAnalysis(models.Model):
    video_file = models.FileField(upload_to='videos/')
    analysis_type = models.CharField(max_length=50, choices=[
        ('object_tracking', 'Object Tracking'),
        ('motion_detection', 'Motion Detection'),
        ('activity_recognition', 'Activity Recognition'),
    ])
    results = models.JSONField(default=dict)
    fps = models.FloatField()
    duration = models.FloatField()
    frame_count = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

class TrackingResult(models.Model):
    video_analysis = models.ForeignKey(VideoAnalysis, on_delete=models.CASCADE)
    object_id = models.CharField(max_length=50)
    object_class = models.CharField(max_length=100)
    trajectory = models.JSONField(default=list)  # List of [frame, x, y, w, h]
    confidence_scores = models.JSONField(default=list)
    first_frame = models.IntegerField()
    last_frame = models.IntegerField()

# Video tracking service
import cv2
from collections import defaultdict
import numpy as np

class VideoTrackingService:
    def __init__(self):
        # Initialize object detector
        self.detector = YOLO('yolov8n.pt')
        
        # Initialize tracker
        self.tracker_type = 'CSRT'  # KCF, CSRT, or MIL
        
    def analyze_video(self, video_path, analysis_type='object_tracking'):
        """Comprehensive video analysis"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        analysis_results = {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'tracks': defaultdict(list),
            'motion_events': [],
            'activities': []
        }
        
        if analysis_type == 'object_tracking':
            analysis_results = self.track_objects(cap, analysis_results)
        elif analysis_type == 'motion_detection':
            analysis_results = self.detect_motion(cap, analysis_results)
        elif analysis_type == 'activity_recognition':
            analysis_results = self.recognize_activities(cap, analysis_results)
        
        cap.release()
        return analysis_results
    
    def track_objects(self, cap, results):
        """Multi-object tracking across video frames"""
        trackers = {}
        track_id_counter = 0
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % 5 == 0:  # Detect every 5th frame for efficiency
                detections = self.detector(frame)
                
                # Process detections
                for detection in detections:
                    boxes = detection.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = self.detector.names[class_id]
                            
                            if confidence > 0.5:
                                # Create new tracker for high-confidence detections
                                bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                                
                                # Initialize tracker
                                if self.tracker_type == 'CSRT':
                                    tracker = cv2.TrackerCSRT_create()
                                elif self.tracker_type == 'KCF':
                                    tracker = cv2.TrackerKCF_create()
                                else:
                                    tracker = cv2.TrackerMIL_create()
                                
                                success = tracker.init(frame, bbox)
                                if success:
                                    track_id = f"{class_name}_{track_id_counter}"
                                    trackers[track_id] = {
                                        'tracker': tracker,
                                        'class': class_name,
                                        'last_seen': frame_num
                                    }
                                    track_id_counter += 1
            
            # Update all trackers
            for track_id, track_data in list(trackers.items()):
                success, bbox = track_data['tracker'].update(frame)
                
                if success:
                    x, y, w, h = bbox
                    results['tracks'][track_id].append({
                        'frame': frame_num,
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'center': [int(x + w/2), int(y + h/2)]
                    })
                    track_data['last_seen'] = frame_num
                else:
                    # Remove failed trackers
                    del trackers[track_id]
            
            frame_num += 1
        
        return results
    
    def detect_motion(self, cap, results):
        """Detect motion events in video"""
        # Background subtraction for motion detection
        back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        frame_num = 0
        motion_threshold = 1000  # Minimum pixels for motion event
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply background subtraction
            fg_mask = back_sub.apply(frame)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            motion_areas = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > motion_threshold:
                    motion_detected = True
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_areas.append([x, y, w, h])
            
            if motion_detected:
                results['motion_events'].append({
                    'frame': frame_num,
                    'timestamp': frame_num / results['fps'],
                    'areas': motion_areas,
                    'total_motion_area': sum(w*h for x,y,w,h in motion_areas)
                })
            
            frame_num += 1
        
        return results
    
    def calculate_trajectory_features(self, trajectory):
        """Calculate motion features from trajectory"""
        if len(trajectory) < 2:
            return {}
        
        # Extract center points
        centers = [point['center'] for point in trajectory]
        
        # Calculate velocity
        velocities = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
        
        # Calculate direction changes
        direction_changes = 0
        for i in range(2, len(centers)):
            # Calculate vectors
            v1 = np.array(centers[i-1]) - np.array(centers[i-2])
            v2 = np.array(centers[i]) - np.array(centers[i-1])
            
            # Calculate angle between vectors
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                if angle > np.pi/4:  # 45 degrees
                    direction_changes += 1
        
        return {
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'max_velocity': np.max(velocities) if velocities else 0,
            'direction_changes': direction_changes,
            'path_length': sum(velocities) if velocities else 0,
            'displacement': np.linalg.norm(
                np.array(centers[-1]) - np.array(centers[0])
            ) if len(centers) > 1 else 0
        }

# views.py
from django.views.generic import View
from django.core.files.storage import default_storage

class VideoAnalysisView(View):
    def __init__(self):
        super().__init__()
        self.tracking_service = VideoTrackingService()
    
    def post(self, request):
        video_file = request.FILES.get('video')
        analysis_type = request.POST.get('analysis_type', 'object_tracking')
        
        if not video_file:
            return JsonResponse({'error': 'No video file provided'}, status=400)
        
        try:
            # Save video file
            file_path = default_storage.save(
                f'temp_videos/{video_file.name}', 
                video_file
            )
            full_path = default_storage.path(file_path)
            
            # Perform analysis
            results = self.tracking_service.analyze_video(full_path, analysis_type)
            
            # Create database record
            analysis = VideoAnalysis.objects.create(
                video_file=video_file,
                analysis_type=analysis_type,
                results=results,
                fps=results['fps'],
                duration=results['duration'],
                frame_count=results['frame_count']
            )
            
            # Create tracking results for each detected track
            for track_id, trajectory in results['tracks'].items():
                if trajectory:
                    # Calculate trajectory features
                    features = self.tracking_service.calculate_trajectory_features(trajectory)
                    
                    TrackingResult.objects.create(
                        video_analysis=analysis,
                        object_id=track_id,
                        object_class=track_id.split('_')[0],
                        trajectory=trajectory,
                        confidence_scores=[0.8] * len(trajectory),  # Placeholder
                        first_frame=trajectory[0]['frame'],
                        last_frame=trajectory[-1]['frame']
                    )
            
            # Clean up temporary file
            default_storage.delete(file_path)
            
            return JsonResponse({
                'success': True,
                'analysis_id': analysis.id,
                'summary': {
                    'total_tracks': len(results['tracks']),
                    'motion_events': len(results.get('motion_events', [])),
                    'duration': results['duration'],
                    'fps': results['fps']
                },
                'tracks': {
                    track_id: {
                        'frames_tracked': len(trajectory),
                        'features': self.tracking_service.calculate_trajectory_features(trajectory)
                    }
                    for track_id, trajectory in results['tracks'].items()
                }
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
```

**Syntax Explanation:**
- `defaultdict(list)`: Creates dictionary that automatically creates empty lists for new keys
- `cv2.TrackerCSRT_create()`: Creates CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) tracker
- `back_sub.apply()`: Applies background subtraction to detect moving objects
- `cv2.findContours()`: Finds object boundaries in binary images
- `np.linalg.norm()`: Calculates Euclidean distance/vector magnitude
- `np.arccos(np.clip())`: Safely calculates angle between vectors with clipping to valid range

---

## Final Project: Intelligent Restaurant Security & Customer Experience System

Now let's combine all our advanced computer vision skills into a comprehensive system that would make any restaurant owner's eyes light up with possibilities.

### The Complete Visual Intelligence Platform

```python
# Final Project: restaurant_vision/models.py
from django.db import models
import json
from datetime import datetime, timedelta
from django.utils import timezone

class RestaurantArea(models.Model):
    name = models.CharField(max_length=100)  # "Entrance", "Dining", "Kitchen", "Bar"
    camera_id = models.CharField(max_length=50)
    is_active = models.BooleanField(default=True)
    
class SecurityEvent(models.Model):
    SEVERITY_CHOICES = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'), 
        ('HIGH', 'High'),
        ('CRITICAL', 'Critical')
    ]
    
    area = models.ForeignKey(RestaurantArea, on_delete=models.CASCADE)
    event_type = models.CharField(max_length=50)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    description = models.TextField()
    video_evidence = models.FileField(upload_to='security_videos/', null=True)
    screenshot = models.ImageField(upload_to='security_screenshots/', null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    resolved = models.BooleanField(default=False)

class CustomerExperience(models.Model):
    customer = models.ForeignKey('CustomerProfile', on_delete=models.CASCADE, null=True)
    area = models.ForeignKey(RestaurantArea, on_delete=models.CASCADE)
    entry_time = models.DateTimeField()
    exit_time = models.DateTimeField(null=True)
    mood_analysis = models.JSONField(default=dict)  # Happy, frustrated, excited, etc.
    waiting_time = models.DurationField(null=True)
    service_interactions = models.JSONField(default=list)

# restaurant_vision/services.py
class IntegratedVisionSystem:
    def __init__(self):
        self.object_detector = YOLO('yolov8n.pt')
        self.face_recognition = FacialRecognitionService()
        self.style_transfer = StyleTransferService()
        self.video_tracker = VideoTrackingService()
        
        # Load emotion detection model (simplified)
        self.emotion_labels = ['happy', 'sad', 'angry', 'surprised', 'neutral', 'frustrated']
        
    def process_live_feed(self, camera_id):
        """Process live camera feed with all vision capabilities"""
        area = RestaurantArea.objects.get(camera_id=camera_id)
        
        # Initialize video capture (assuming IP camera or webcam)
        cap = cv2.VideoCapture(f'rtmp://camera-{camera_id}.local/stream')
        
        frame_count = 0
        active_tracks = {}
        customer_sessions = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = timezone.now()
            
            # Object Detection - identify people, objects, potential issues
            detections = self.object_detector(frame)
            
            people_detected = []
            security_objects = []
            
            for detection in detections:
                boxes = detection.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.object_detector.names[class_id]
                        
                        if confidence > 0.6:
                            if class_name == 'person':
                                people_detected.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(confidence)
                                })
                            elif class_name in ['knife', 'bottle', 'cell phone', 'handbag']:
                                security_objects.append({
                                    'object': class_name,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(confidence)
                                })
            
            # Face Recognition for each detected person
            for person_bbox in people_detected:
                x1, y1, x2, y2 = person_bbox['bbox']
                person_crop = frame[y1:y2, x1:x2]
                
                # Save temporary crop for face recognition
                temp_path = f'/tmp/person_crop_{frame_count}.jpg'
                cv2.imwrite(temp_path, person_crop)
                
                try:
                    face_results = self.face_recognition.recognize_face(temp_path)
                    
                    for face_result in face_results:
                        if face_result['name'] != 'Unknown':
                            # Known customer detected
                            customer = CustomerProfile.objects.get(name=face_result['name'])
                            
                            # Track customer experience
                            session_key = f"{customer.id}_{area.id}"
                            if session_key not in customer_sessions:
                                customer_sessions[session_key] = CustomerExperience.objects.create(
                                    customer=customer,
                                    area=area,
                                    entry_time=current_time
                                )
                            
                            # Analyze customer mood (simplified emotion detection)
                            emotion = self.analyze_customer_emotion(person_crop)
                            if emotion:
                                experience = customer_sessions[session_key]
                                mood_data = experience.mood_analysis
                                mood_data[str(current_time.timestamp())] = emotion
                                experience.mood_analysis = mood_data
                                experience.save()
                        else:
                            # Unknown person - potential security consideration
                            if area.name in ['Kitchen', 'Staff Only']:
                                self.create_security_event(
                                    area,
                                    'UNAUTHORIZED_ACCESS',
                                    'MEDIUM',
                                    f'Unknown person detected in restricted area: {area.name}',
                                    frame
                                )
                
                except Exception as e:
                    print(f"Face recognition error: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            # Security Analysis
            self.analyze_security_situation(area, security_objects, people_detected, frame)
            
            # Activity Recognition (simplified)
            if len(people_detected) > 0:
                activity = self.recognize_activity_pattern(people_detected, active_tracks)
                if activity['risk_level'] > 0.7:
                    self.create_security_event(
                        area,
                        'SUSPICIOUS_ACTIVITY',
                        'HIGH',
                        f'Suspicious activity detected: {activity["description"]}',
                        frame
                    )
            
            frame_count += 1
            
            # Process every 30 frames (approximately 1 second at 30fps)
            if frame_count % 30 == 0:
                self.cleanup_old_sessions(customer_sessions, current_time)
            
            # Break after processing (in real implementation, this would run continuously)
            if frame_count > 1000:  # Process 1000 frames for demo
                break
        
        cap.release()
        return self.generate_analysis_report(area, current_time)
    
    def analyze_customer_emotion(self, face_crop):
        """Simplified emotion analysis"""
        # In a real implementation, you'd use a trained emotion detection model
        # For demo purposes, we'll return a random emotion with some logic
        height, width = face_crop.shape[:2]
        
        # Simple heuristics based on image properties
        brightness = np.mean(face_crop)
        
        if brightness > 150:
            return {'emotion': 'happy', 'confidence': 0.8}
        elif brightness < 100:
            return {'emotion': 'neutral', 'confidence': 0.6}
        else:
            return {'emotion': 'neutral', 'confidence': 0.5}
    
    def analyze_security_situation(self, area, security_objects, people_count, frame):
        """Analyze current frame for security concerns"""
        current_time = timezone.now()
        
        # Check for overcrowding
        if len(people_count) > 20 and area.name == 'Dining':
            self.create_security_event(
                area,
                'OVERCROWDING',
                'MEDIUM',
                f'Overcrowding detected: {len(people_count)} people in dining area',
                frame
            )
        
        # Check for weapons or dangerous objects
        dangerous_objects = [obj for obj in security_objects if obj['object'] == 'knife']
        if dangerous_objects and area.name != 'Kitchen':
            self.create_security_event(
                area,
                'WEAPON_DETECTED',
                'CRITICAL',
                f'Knife detected in {area.name} area',
                frame
            )
        
        # Check for after-hours activity
        if current_time.hour < 6 or current_time.hour > 23:
            if len(people_count) > 0:
                self.create_security_event(
                    area,
                    'AFTER_HOURS_ACTIVITY',
                    'HIGH',
                    f'Activity detected during closed hours: {len(people_count)} people',
                    frame
                )
    
    def create_security_event(self, area, event_type, severity, description, frame):
        """Create and log security event"""
        # Save screenshot
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = f'security_screenshots/{area.name}_{timestamp}.jpg'
        
        # Save frame as screenshot
        cv2.imwrite(f'/tmp/{screenshot_path}', frame)
        
        # Create security event
        event = SecurityEvent.objects.create(
            area=area,
            event_type=event_type,
            severity=severity,
            description=description,
        )
        
        # In a real system, you'd also:
        # 1. Send immediate alerts to security personnel
        # 2. Save video clip of the event
        # 3. Trigger automated responses (lights, alarms, etc.)
        
        print(f"SECURITY ALERT: {severity} - {description}")
        return event
    
    def recognize_activity_pattern(self, people_detections, active_tracks):
        """Simplified activity recognition"""
        # Analyze movement patterns, grouping, etc.
        person_count = len(people_detections)
        
        # Simple risk assessment
        risk_factors = {
            'large_group': person_count > 10,
            'unusual_time': timezone.now().hour < 7 or timezone.now().hour > 22,
            'restricted_area': False  # Would be determined by area type
        }
        
        risk_level = sum(risk_factors.values()) / len(risk_factors)
        
        return {
            'risk_level': risk_level,
            'description': f'Group of {person_count} people detected',
            'factors': risk_factors
        }
    
    def cleanup_old_sessions(self, customer_sessions, current_time):
        """Clean up customer sessions that have ended"""
        for session_key, session in list(customer_sessions.items()):
            # If no update for 5 minutes, consider session ended
            if current_time - session.entry_time > timedelta(minutes=5):
                session.exit_time = current_time
                session.waiting_time = session.exit_time - session.entry_time
                session.save()
                del customer_sessions[session_key]
    
    def generate_analysis_report(self, area, end_time):
        """Generate comprehensive analysis report"""
        start_time = end_time - timedelta(hours=1)  # Last hour
        
        # Get recent events
        recent_events = SecurityEvent.objects.filter(
            area=area,
            timestamp__gte=start_time
        )
        
        recent_experiences = CustomerExperience.objects.filter(
            area=area,
            entry_time__gte=start_time
        )
        
        # Calculate statistics
        total_visitors = recent_experiences.count()
        avg_visit_duration = recent_experiences.aggregate(
            avg_duration=models.Avg('waiting_time')
        )['avg_duration']
        
        security_incidents = recent_events.count()
        critical_incidents = recent_events.filter(severity='CRITICAL').count()
        
        return {
            'area': area.name,
            'time_period': f"{start_time} to {end_time}",
            'visitor_stats': {
                'total_visitors': total_visitors,
                'average_visit_duration': str(avg_visit_duration) if avg_visit_duration else 'N/A',
                'returning_customers': recent_experiences.filter(customer__isnull=False).count()
            },
            'security_stats': {
                'total_incidents': security_incidents,
                'critical_incidents': critical_incidents,
                'incident_types': list(recent_events.values_list('event_type', flat=True).distinct())
            },
            'recommendations': self.generate_recommendations(recent_events, recent_experiences)
        }
    
    def generate_recommendations(self, events, experiences):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Security recommendations
        if events.filter(event_type='OVERCROWDING').exists():
            recommendations.append("Consider implementing crowd control measures during peak hours")
        
        if events.filter(event_type='UNAUTHORIZED_ACCESS').exists():
            recommendations.append("Review access control for restricted areas")
        
        # Customer experience recommendations
        long_waits = experiences.filter(waiting_time__gt=timedelta(minutes=15)).count()
        if long_waits > 0:
            recommendations.append(f"{long_waits} customers waited over 15 minutes - consider optimizing service flow")
        
        # Mood analysis
        negative_moods = 0
        for exp in experiences:
            for timestamp, mood in exp.mood_analysis.items():
                if mood.get('emotion') in ['frustrated', 'angry', 'sad']:
                    negative_moods += 1
        
        if negative_moods > experiences.count() * 0.3:  # More than 30% negative
            recommendations.append("High negative emotion detection - investigate service quality issues")
        
        return recommendations

# restaurant_vision/views.py
from django.views.generic import TemplateView
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

class RestaurantDashboardView(TemplateView):
    template_name = 'restaurant_vision/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get all areas
        context['areas'] = RestaurantArea.objects.filter(is_active=True)
        
        # Get recent security events
        context['recent_events'] = SecurityEvent.objects.filter(
            timestamp__gte=timezone.now() - timedelta(hours=24)
        ).order_by('-timestamp')[:10]
        
        # Get current customer experiences
        context['active_sessions'] = CustomerExperience.objects.filter(
            exit_time__isnull=True
        )
        
        return context

@method_decorator(csrf_exempt, name='dispatch')
class LiveAnalysisView(View):
    def __init__(self):
        super().__init__()
        self.vision_system = IntegratedVisionSystem()
    
    def post(self, request):
        camera_id = request.POST.get('camera_id')
        
        if not camera_id:
            return JsonResponse({'error': 'Camera ID required'}, status=400)
        
        try:
            # Start live analysis
            report = self.vision_system.process_live_feed(camera_id)
            
            return JsonResponse({
                'success': True,
                'analysis_report': report
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

# restaurant_vision/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.RestaurantDashboardView.as_view(), name='dashboard'),
    path('analyze/', views.LiveAnalysisView.as_view(), name='live_analysis'),
]
```

**Final Syntax Explanation:**
- `timezone.now()`: Django's timezone-aware datetime function
- `timedelta(minutes=5)`: Creates time duration objects for calculations
- `models.Avg()`: Django ORM aggregation function for calculating averages
- `values_list('field', flat=True)`: Returns flat list of specific field values
- `@method_decorator(csrf_exempt)`: Applies CSRF exemption to class-based views

---

# Real-time Object Detection System

## Project Overview
Build a comprehensive real-time object detection system that can identify and track multiple objects through a webcam feed or uploaded videos. This system combines computer vision techniques with a Django web interface for a complete end-to-end solution.

## Project Structure
```
object_detection_system/
├── manage.py
├── requirements.txt
├── detection_project/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── detector/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── object_detector.py
│   │   └── video_processor.py
│   └── templates/
│       └── detector/
│           ├── index.html
│           ├── upload.html
│           └── results.html
├── media/
│   ├── uploads/
│   └── processed/
└── static/
    ├── css/
    ├── js/
    └── models/
```

## Core Detection Engine

### object_detector.py
```python
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class RealTimeObjectDetector:
    def __init__(self, model_path='static/models/yolo'):
        """
        Initialize the object detector with YOLO model
        Like setting up your main cooking station with all essential tools
        """
        self.model_path = Path(model_path)
        self.net = None
        self.output_layers = None
        self.classes = []
        self.colors = np.random.uniform(0, 255, size=(80, 3))
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.load_model()
    
    def load_model(self):
        """Load YOLO model and configuration"""
        try:
            # Load YOLO network
            weights_path = self.model_path / 'yolov4.weights'
            config_path = self.model_path / 'yolov4.cfg'
            
            self.net = cv2.dnn.readNet(str(weights_path), str(config_path))
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Load class names
            with open(self.model_path / 'coco.names', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
                
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to OpenCV's built-in cascade for demo
            self.use_cascade_fallback()
    
    def use_cascade_fallback(self):
        """Fallback detection method using Haar cascades"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    
    def detect_objects(self, frame):
        """
        Detect objects in a single frame
        Like a chef quickly identifying all ingredients on the counter
        """
        height, width, channels = frame.shape
        detections = []
        
        if self.net is not None:
            # YOLO detection
            detections = self._yolo_detection(frame, width, height)
        else:
            # Cascade detection fallback
            detections = self._cascade_detection(frame)
        
        return detections
    
    def _yolo_detection(self, frame, width, height):
        """Perform YOLO-based object detection"""
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each detection
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                class_name = self.classes[class_ids[i]]
                confidence = confidences[i]
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x, y, w, h],
                    'center': [x + w//2, y + h//2]
                })
        
        return detections
    
    def _cascade_detection(self, frame):
        """Fallback cascade detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        bodies = self.body_cascade.detectMultiScale(gray, 1.1, 4)
        
        detections = []
        
        # Process faces
        for (x, y, w, h) in faces:
            detections.append({
                'class': 'face',
                'confidence': 0.8,
                'bbox': [x, y, w, h],
                'center': [x + w//2, y + h//2]
            })
        
        # Process bodies
        for (x, y, w, h) in bodies:
            detections.append({
                'class': 'person',
                'confidence': 0.7,
                'bbox': [x, y, w, h],
                'center': [x + w//2, y + h//2]
            })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        Like a chef plating and labeling each dish
        """
        for detection in detections:
            x, y, w, h = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Choose color based on class
            if self.net is not None and class_name in self.classes:
                color = self.colors[self.classes.index(class_name)]
            else:
                color = (0, 255, 0)  # Default green
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with confidence
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background rectangle for text
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame

class ObjectTracker:
    """Advanced object tracking across frames"""
    
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def register(self, centroid):
        """Register a new object"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        """
        Update object tracking with new detections
        Like a chef keeping track of multiple dishes cooking simultaneously
        """
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
        # Extract centroids from detections
        input_centroids = np.array([det['center'] for det in detections])
        
        if len(self.objects) == 0:
            # Register all detections as new objects
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Match existing objects to new detections
            object_centroids = np.array(list(self.objects.values()))
            
            # Compute distance matrix
            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
            
            # Find minimum distances
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indexes = set()
            used_col_indexes = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indexes or col in used_col_indexes:
                    continue
                
                if D[row, col] > 50:  # Maximum distance threshold
                    continue
                
                # Update object position
                object_id = list(self.objects.keys())[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_row_indexes.add(row)
                used_col_indexes.add(col)
            
            # Handle unmatched detections and objects
            unused_rows = set(range(0, D.shape[0])).difference(used_row_indexes)
            unused_cols = set(range(0, D.shape[1])).difference(used_col_indexes)
            
            if D.shape[0] >= D.shape[1]:
                # More existing objects than detections
                for row in unused_rows:
                    object_id = list(self.objects.keys())[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # More detections than existing objects
                for col in unused_cols:
                    self.register(input_centroids[col])
        
        # Return tracking results with detection data
        tracking_results = {}
        for i, detection in enumerate(detections):
            centroid = detection['center']
            # Find closest tracked object
            min_distance = float('inf')
            object_id = None
            
            for oid, obj_centroid in self.objects.items():
                distance = np.linalg.norm(np.array(centroid) - np.array(obj_centroid))
                if distance < min_distance:
                    min_distance = distance
                    object_id = oid
            
            if object_id is not None and min_distance < 50:
                tracking_results[object_id] = {
                    **detection,
                    'object_id': object_id,
                    'tracking_confidence': max(0, 1 - min_distance/50)
                }
        
        return tracking_results
```

### video_processor.py
```python
import cv2
import os
from django.conf import settings
from .object_detector import RealTimeObjectDetector, ObjectTracker
import json
from datetime import datetime

class VideoProcessor:
    def __init__(self):
        """
        Initialize video processor
        Like setting up a complete cooking station for video preparation
        """
        self.detector = RealTimeObjectDetector()
        self.tracker = ObjectTracker()
        self.processed_frames = []
        
    def process_webcam_stream(self):
        """
        Process live webcam stream
        Like preparing fresh ingredients as they arrive
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise Exception("Could not open webcam")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect objects
                detections = self.detector.detect_objects(frame)
                
                # Update tracking
                tracked_objects = self.tracker.update(detections)
                
                # Draw results
                result_frame = self.detector.draw_detections(frame, list(tracked_objects.values()))
                
                # Add tracking IDs
                for obj_id, obj_data in tracked_objects.items():
                    x, y, w, h = obj_data['bbox']
                    cv2.putText(result_frame, f"ID: {obj_id}", 
                               (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 0, 0), 2)
                
                # Display frame
                cv2.imshow('Real-time Object Detection', result_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def process_uploaded_video(self, video_path, output_path):
        """
        Process uploaded video file
        Like carefully preparing a pre-selected set of ingredients
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_log = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                detections = self.detector.detect_objects(frame)
                tracked_objects = self.tracker.update(detections)
                
                # Log detection data
                frame_data = {
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'detections': len(detections),
                    'tracked_objects': len(tracked_objects),
                    'objects': []
                }
                
                for obj_id, obj_data in tracked_objects.items():
                    frame_data['objects'].append({
                        'id': obj_id,
                        'class': obj_data['class'],
                        'confidence': obj_data['confidence'],
                        'bbox': obj_data['bbox']
                    })
                
                detection_log.append(frame_data)
                
                # Draw results
                result_frame = self.detector.draw_detections(frame, list(tracked_objects.values()))
                
                # Add tracking information
                for obj_id, obj_data in tracked_objects.items():
                    x, y, w, h = obj_data['bbox']
                    cv2.putText(result_frame, f"ID: {obj_id}", 
                               (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 0, 0), 2)
                
                # Write frame
                out.write(result_frame)
                frame_count += 1
                
                # Progress callback could be added here
                if frame_count % 30 == 0:  # Every second at 30fps
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing: {progress:.1f}%")
                    
        finally:
            cap.release()
            out.release()
        
        # Save detection log
        log_path = output_path.replace('.mp4', '_detections.json')
        with open(log_path, 'w') as f:
            json.dump(detection_log, f, indent=2)
        
        return {
            'output_video': output_path,
            'detection_log': log_path,
            'total_frames': frame_count,
            'duration': frame_count / fps
        }
    
    def generate_analytics(self, detection_log_path):
        """
        Generate analytics from detection log
        Like analyzing the cooking process to improve next time
        """
        with open(detection_log_path, 'r') as f:
            log_data = json.load(f)
        
        analytics = {
            'total_frames': len(log_data),
            'total_detections': sum(frame['detections'] for frame in log_data),
            'unique_objects': set(),
            'object_classes': {},
            'detection_timeline': [],
            'confidence_stats': {
                'min': 1.0,
                'max': 0.0,
                'avg': 0.0
            }
        }
        
        all_confidences = []
        
        for frame in log_data:
            frame_classes = []
            for obj in frame['objects']:
                analytics['unique_objects'].add(obj['id'])
                class_name = obj['class']
                
                if class_name not in analytics['object_classes']:
                    analytics['object_classes'][class_name] = 0
                analytics['object_classes'][class_name] += 1
                
                frame_classes.append(class_name)
                all_confidences.append(obj['confidence'])
            
            analytics['detection_timeline'].append({
                'timestamp': frame['timestamp'],
                'count': len(frame['objects']),
                'classes': list(set(frame_classes))
            })
        
        # Calculate confidence statistics
        if all_confidences:
            analytics['confidence_stats']['min'] = min(all_confidences)
            analytics['confidence_stats']['max'] = max(all_confidences)
            analytics['confidence_stats']['avg'] = sum(all_confidences) / len(all_confidences)
        
        analytics['unique_objects'] = len(analytics['unique_objects'])
        
        return analytics

class LiveStreamProcessor:
    """Handle real-time streaming with web interface"""
    
    def __init__(self):
        self.detector = RealTimeObjectDetector()
        self.tracker = ObjectTracker()
        self.is_streaming = False
        
    def start_stream(self):
        """Start live detection stream"""
        self.is_streaming = True
        cap = cv2.VideoCapture(0)
        
        while self.is_streaming and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Process frame
            detections = self.detector.detect_objects(frame)
            tracked_objects = self.tracker.update(detections)
            
            # Create result frame
            result_frame = self.detector.draw_detections(frame, list(tracked_objects.values()))
            
            # Convert to JPEG for streaming
            _, buffer = cv2.imencode('.jpg', result_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()
    
    def stop_stream(self):
        """Stop the live stream"""
        self.is_streaming = False
```

## Django Integration

### models.py
```python
from django.db import models
from django.contrib.auth.models import User
import json

class DetectionSession(models.Model):
    """
    Store detection session data
    Like keeping a record of every cooking session
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_name = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    session_type = models.CharField(max_length=20, choices=[
        ('webcam', 'Webcam Stream'),
        ('video', 'Video Upload'),
        ('image', 'Image Upload')
    ])
    
    # File paths
    input_file = models.FileField(upload_to='uploads/', null=True, blank=True)
    output_file = models.FileField(upload_to='processed/', null=True, blank=True)
    
    # Detection results
    total_detections = models.IntegerField(default=0)
    unique_objects = models.IntegerField(default=0)
    processing_time = models.FloatField(null=True, blank=True)  # in seconds
    
    # Analytics data (stored as JSON)
    analytics_data = models.TextField(null=True, blank=True)
    
    def set_analytics(self, data):
        self.analytics_data = json.dumps(data)
    
    def get_analytics(self):
        if self.analytics_data:
            return json.loads(self.analytics_data)
        return {}
    
    class Meta:
        ordering = ['-created_at']

class DetectedObject(models.Model):
    """
    Individual detected objects
    Like cataloging each ingredient found in the kitchen
    """
    session = models.ForeignKey(DetectionSession, related_name='detected_objects', on_delete=models.CASCADE)
    object_class = models.CharField(max_length=100)
    confidence = models.FloatField()
    
    # Bounding box coordinates
    bbox_x = models.IntegerField()
    bbox_y = models.IntegerField()
    bbox_width = models.IntegerField()
    bbox_height = models.IntegerField()
    
    # Tracking information
    object_id = models.IntegerField(null=True, blank=True)
    frame_number = models.IntegerField(default=0)
    timestamp = models.FloatField(default=0.0)  # Time in video
    
    first_seen = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['frame_number', 'object_id']
```

### views.py
```python
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import json
import time
from datetime import datetime

from .models import DetectionSession, DetectedObject
from .utils.video_processor import VideoProcessor, LiveStreamProcessor
from .utils.object_detector import RealTimeObjectDetector

def index(request):
    """
    Main dashboard view
    Like the main cooking area where everything begins
    """
    recent_sessions = DetectionSession.objects.all()[:10]
    
    # Calculate total statistics
    total_sessions = DetectionSession.objects.count()
    total_detections = sum(session.total_detections for session in DetectionSession.objects.all())
    
    context = {
        'recent_sessions': recent_sessions,
        'total_sessions': total_sessions,
        'total_detections': total_detections,
    }
    
    return render(request, 'detector/index.html', context)

def upload_video(request):
    """
    Handle video upload and processing
    Like preparing ingredients for a complex dish
    """
    if request.method == 'POST':
        if 'video_file' not in request.FILES:
            messages.error(request, 'No video file selected')
            return redirect('detector:upload')
        
        video_file = request.FILES['video_file']
        session_name = request.POST.get('session_name', f'Video_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Create detection session
        session = DetectionSession.objects.create(
            user=request.user if request.user.is_authenticated else None,
            session_name=session_name,
            session_type='video',
            input_file=video_file
        )
        
        # Save uploaded file
        input_path = os.path.join(settings.MEDIA_ROOT, 'uploads', video_file.name)
        output_filename = f"processed_{session.id}_{video_file.name}"
        output_path = os.path.join(settings.MEDIA_ROOT, 'processed', output_filename)
        
        try:
            # Process video
            processor = VideoProcessor()
            start_time = time.time()
            
            result = processor.process_uploaded_video(input_path, output_path)
            
            processing_time = time.time() - start_time
            
            # Update session with results
            session.output_file = f'processed/{output_filename}'
            session.processing_time = processing_time
            session.save()
            
            # Generate and save analytics
            analytics = processor.generate_analytics(result['detection_log'])
            session.set_analytics(analytics)
            session.total_detections = analytics['total_detections']
            session.unique_objects = analytics['unique_objects']
            session.save()
            
            # Save individual detections to database
            with open(result['detection_log'], 'r') as f:
                log_data = json.load(f)
            
            for frame_data in log_data:
                for obj in frame_data['objects']:
                    DetectedObject.objects.create(
                        session=session,
                        object_class=obj['class'],
                        confidence=obj['confidence'],
                        bbox_x=obj['bbox'][0],
                        bbox_y=obj['bbox'][1],
                        bbox_width=obj['bbox'][2],
                        bbox_height=obj['bbox'][3],
                        object_id=obj['id'],
                        frame_number=frame_data['frame_number'],
                        timestamp=frame_data['timestamp']
                    )
            
            messages.success(request, f'Video processed successfully! Found {analytics["total_detections"]} objects.')
            return redirect('detector:session_detail', session_id=session.id)
            
        except Exception as e:
            messages.error(request, f'Error processing video: {str(e)}')
            session.delete()
            return redirect('detector:upload')
    
    return render(request, 'detector/upload.html')

def session_detail(request, session_id):
    """
    Display detailed results for a detection session
    Like presenting the final prepared dish with all details
    """
    session = get_object_or_404(DetectionSession, id=session_id)
    analytics = session.get_analytics()
    
    # Get detection timeline for charts
    detected_objects = session.detected_objects.all()
    
    # Group objects by class for statistics
    class_counts = {}
    for obj in detected_objects:
        if obj.object_class not in class_counts:
            class_counts[obj.object_class] = 0
        class_counts[obj.object_class] += 1
    
    context = {
        'session': session,
        'analytics': analytics,
        'detected_objects': detected_objects[:50],  # Show first 50 for performance
        'class_counts': class_counts,
        'total_objects': detected_objects.count(),
    }
    
    return render(request, 'detector/session_detail.html', context)

def live_stream(request):
    """Render live stream page"""
    return render(request, 'detector/live_stream.html')

def video_feed(request):
    """
    Generate live video feed
    Like serving fresh dishes as they're prepared
    """
    def generate():
        stream_processor = LiveStreamProcessor()
        try:
            yield from stream_processor.start_stream()
        except Exception as e:
            print(f"Streaming error: {e}")
        finally:
            stream_processor.stop_stream()
    
    return StreamingHttpResponse(
        generate(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

@csrf_exempt
def api_detect_image(request):
    """
    API endpoint for single image detection
    Like a quick taste test of a single ingredient
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image file provided'}, status=400)
    
    try:
        image_file = request.FILES['image']
        
        # Save temporary image
        temp_path = default_storage.save(f'temp/{image_file.name}', ContentFile(image_file.read()))
        full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
        
        # Process image
        detector = RealTimeObjectDetector()
        import cv2
        image = cv2.imread(full_path)
        
        if image is None:
            return JsonResponse({'error': 'Invalid image file'}, status=400)
        
        detections = detector.detect_objects(image)
        
        # Clean up temporary file
        os.remove(full_path)
        
        # Format response
        response_data = {
            'detections': len(detections),
            'objects': []
        }
        
        for detection in detections:
            response_data['objects'].append({
                'class': detection['class'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox']
            })
        
        return JsonResponse(response_data)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def analytics_dashboard(request):
    """
    Display comprehensive analytics
    Like reviewing all cooking sessions to improve techniques
    """
    # Get all sessions
    sessions = DetectionSession.objects.all()
    
    # Calculate overall statistics
    total_sessions = sessions.count()
    total_processing_time = sum(s.processing_time or 0 for s in sessions)
    total_detections = sum(s.total_detections for s in sessions)
    
    # Most common object classes
    all_objects = DetectedObject.objects.all()
    class_stats = {}
    
    for obj in all_objects:
        if obj.object_class not in class_stats:
            class_stats[obj.object_class] = {
                'count': 0,
                'avg_confidence': 0,
                'confidences': []
            }
        class_stats[obj.object_class]['count'] += 1
        class_stats[obj.object_class]['confidences'].append(obj.confidence)
    
    # Calculate average confidences
    for class_name, stats in class_stats.items():
        if stats['confidences']:
            stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
    
    # Sort by count
    sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    context = {
        'total_sessions': total_sessions,
        'total_processing_time': total_processing_time,
        'total_detections': total_detections,
        'avg_detections_per_session': total_detections / max(total_sessions, 1),
        'class_stats': sorted_classes[:10],  # Top 10 classes
        'recent_sessions': sessions[:5],
    }
    
    return render(request, 'detector/analytics.html', context)
```

### urls.py
```python
from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_video, name='upload'),
    path('session/<int:session_id>/', views.session_detail, name='session_detail'),
    path('live/', views.live_stream, name='live_stream'),
    path('feed/', views.video_feed, name='video_feed'),
    path('api/detect/', views.api_detect_image, name='api_detect'),
    path('analytics/', views.analytics_dashboard, name='analytics'),
]
```

## Frontend Templates

### templates/detector/index.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Object Detection System</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .stat-label {
            color: #666;
            font-size: 1.1em;
        }
        
        .action-buttons {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .action-btn {
            display: block;
            padding: 20px;
            background: rgba(255, 255, 255, 0.95);
            color: #333;
            text-decoration: none;
            border-radius: 15px;
            text-align: center;
            font-weight: bold;
            font-size: 1.1em;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .action-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            background: #667eea;
            color: white;
        }
        
        .recent-sessions {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .session-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 0;
            border-bottom: 1px solid #eee;
        }
        
        .session-item:last-child {
            border-bottom: none;
        }
        
        .session-info h4 {
            margin: 0 0 5px 0;
            color: #333;
        }
        
        .session-meta {
            color: #666;
            font-size: 0.9em;
        }
        
        .session-stats {
            text-align: right;
            color: #667eea;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Object Detection Hub</h1>
            <p>Real-time AI-powered object detection and tracking system</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ total_sessions }}</div>
                <div class="stat-label">Detection Sessions</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ total_detections }}</div>
                <div class="stat-label">Objects Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{% if total_sessions > 0 %}{{ total_detections|floatformat:0|div:total_sessions }}{% else %}0{% endif %}</div>
                <div class="stat-label">Avg per Session</div>
            </div>
        </div>
        
        <div class="action-buttons">
            <a href="{% url 'detector:live_stream' %}" class="action-btn">
                📹 Live Detection
            </a>
            <a href="{% url 'detector:upload' %}" class="action-btn">
                📁 Upload Video
            </a>
            <a href="{% url 'detector:analytics' %}" class="action-btn">
                📊 Analytics Dashboard
            </a>
        </div>
        
        {% if recent_sessions %}
        <div class="recent-sessions">
            <h2 style="margin-top: 0; color: #333;">Recent Detection Sessions</h2>
            {% for session in recent_sessions %}
            <div class="session-item">
                <div class="session-info">
                    <h4>{{ session.session_name }}</h4>
                    <div class="session-meta">
                        {{ session.created_at|date:"M d, Y H:i" }} • {{ session.get_session_type_display }}
                    </div>
                </div>
                <div class="session-stats">
                    {{ session.total_detections }} objects<br>
                    <a href="{% url 'detector:session_detail' session.id %}" style="color: #667eea; text-decoration: none;">View Details →</a>
                </div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
</body>
</html>
```

### templates/detector/live_stream.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Object Detection</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .video-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }
        
        .video-stream {
            width: 100%;
            max-width: 640px;
            height: auto;
            border-radius: 10px;
            display: block;
            margin: 0 auto;
        }
        
        .controls {
            text-align: center;
            margin-top: 20px;
        }
        
        .control-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            margin: 0 10px;
            transition: all 0.3s ease;
        }
        
        .control-btn:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }
        
        .control-btn:disabled {
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
        }
        
        .info-panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .back-btn {
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 20px;
            text-decoration: none;
            font-weight: bold;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .back-btn:hover {
            background: rgba(255, 255, 255, 0.3);
        }
    </style>
</head>
<body>
    <a href="{% url 'detector:index' %}" class="back-btn">← Back to Home</a>
    
    <div class="container">
        <div class="header">
            <h1>🎥 Live Object Detection</h1>
            <p>Real-time detection from your webcam</p>
        </div>
        
        <div class="video-container">
            <img src="{% url 'detector:video_feed' %}" class="video-stream" alt="Live Detection Feed">
            
            <div class="controls">
                <button class="control-btn" onclick="refreshStream()">🔄 Refresh</button>
                <button class="control-btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
            </div>
        </div>
        
        <div class="info-panel">
            <h3>🎯 How It Works</h3>
            <p>This system processes your webcam feed in real-time, detecting and tracking objects as they appear. Each detected object gets a bounding box with its classification and confidence score.</p>
            
            <h4>🔧 Features:</h4>
            <ul>
                <li><strong>Real-time Detection:</strong> Processes video frames as they arrive</li>
                <li><strong>Object Tracking:</strong> Maintains consistent IDs for moving objects</li>
                <li><strong>Multiple Classes:</strong> Detects people, vehicles, animals, and everyday objects</li>
                <li><strong>Confidence Scoring:</strong> Shows detection certainty for each object</li>
            </ul>
            
            <p><em>Note: Make sure to allow camera access when prompted by your browser.</em></p>
        </div>
    </div>
    
    <script>
        function refreshStream() {
            const img = document.querySelector('.video-stream');
            const src = img.src;
            img.src = '';
            setTimeout(() => {
                img.src = src + '?t=' + new Date().getTime();
            }, 100);
        }
        
        function toggleFullscreen() {
            const img = document.querySelector('.video-stream');
            if (!document.fullscreenElement) {
                img.requestFullscreen().catch(err => {
                    console.log('Error attempting to enable fullscreen:', err.message);
                });
            } else {
                document.exitFullscreen();
            }
        }
        
        // Auto-refresh stream every 30 seconds to prevent timeouts
        setInterval(refreshStream, 30000);
    </script>
</body>
</html>
```

## Configuration Files

### requirements.txt
```
Django==4.2.7
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.0.1
django-extensions==3.2.3
```

### settings.py additions
```python
import os
from pathlib import Path

# Add to INSTALLED_APPS
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'detector',  # Our detection app
]

# Media files configuration
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Create media directories
os.makedirs(os.path.join(MEDIA_ROOT, 'uploads'), exist_ok=True)
os.makedirs(os.path.join(MEDIA_ROOT, 'processed'), exist_ok=True)

# Static files
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

# Create static directories
os.makedirs(os.path.join(BASE_DIR, 'static', 'models'), exist_ok=True)
```

## Deployment Script

### deploy.py
```python
#!/usr/bin/env python3
"""
Deployment script for Real-time Object Detection System
Like preparing the entire kitchen for service
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e.stderr}")
        return None

def setup_project():
    """Set up the Django project"""
    print("🚀 Setting up Real-time Object Detection System")
    
    # Create project structure
    commands = [
        ("python -m venv detection_env", "Creating virtual environment"),
        ("source detection_env/bin/activate || detection_env\\Scripts\\activate", "Activating virtual environment"),
        ("pip install -r requirements.txt", "Installing Python dependencies"),
        ("python manage.py makemigrations", "Creating database migrations"),
        ("python manage.py migrate", "Running database migrations"),
        ("python manage.py collectstatic --noinput", "Collecting static files"),
    ]
    
    for command, description in commands:
        result = run_command(command, description)
        if result is None:
            print("❌ Setup failed. Please check the errors above.")
            return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Download YOLO model files to static/models/ directory")
    print("2. Run: python manage.py runserver")
    print("3. Open http://localhost:8000 in your browser")
    
    return True

def download_models():
    """Download required model files"""
    print("📥 Downloading YOLO model files...")
    
    model_dir = Path("static/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for YOLO files (replace with actual URLs)
    model_files = {
        "yolov4.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
        "yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
        "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
    }
    
    for filename, url in model_files.items():
        filepath = model_dir / filename
        if not filepath.exists():
            command = f"wget -O {filepath} {url}"
            run_command(command, f"Downloading {filename}")
    
    print("✅ Model files ready")

if __name__ == "__main__":
    if "--models" in sys.argv:
        download_models()
    else:
        setup_project()
```

## Usage Examples

### Running the System
```bash
# 1. Set up the project
python deploy.py

# 2. Download model files (optional - system works with fallback)
python deploy.py --models

# 3. Start the development server
python manage.py runserver

# 4. Access the web interface
# Open http://localhost:8000 in your browser
```

### API Usage Example
```python
import requests

# Detect objects in an image via API
url = 'http://localhost:8000/api/detect/'
files = {'image': open('test_image.jpg', 'rb')}

response = requests.post(url, files=files)
result = response.json()

print(f"Found {result['detections']} objects:")
for obj in result['objects']:
    print(f"- {obj['class']}: {obj['confidence']:.2f}")
```

This real-time object detection system combines advanced computer vision techniques with Django's web framework to create a comprehensive solution. Like a well-equipped kitchen that can handle any cooking challenge, this system provides multiple ways to detect and analyze objects - from live webcam feeds to uploaded videos, complete with tracking, analytics, and a user-friendly web interface.

## Assignment: Smart Retail Analytics System

Create a computer vision system for a retail store that combines object detection, customer tracking, and behavioral analysis.

**Requirements:**

1. **Product Detection**: Detect when products are picked up, examined, or returned to shelves
2. **Customer Journey Mapping**: Track customer movement patterns through store sections  
3. **Queue Analysis**: Monitor checkout line lengths and waiting times
4. **Shelf Monitoring**: Detect when shelves need restocking based on visual analysis
5. **Heat Map Generation**: Create visual heat maps showing popular store areas

**Your system should:**

- Use Django models to store customer journeys, product interactions, and store analytics
- Implement real-time video processing with object tracking
- Generate daily/weekly reports with actionable insights
- Include a simple web dashboard showing live metrics
- Handle at least 3 different camera feeds simultaneously

**Deliverables:**

1. Django models for storing analytics data
2. Video processing service that combines detection + tracking
3. Analytics calculation methods (dwell time, conversion rates, traffic patterns)
4. Simple HTML dashboard with charts/statistics
5. README explaining your approach and key insights discovered

**Evaluation Criteria:**

- Code organization and Django best practices
- Accuracy of computer vision implementations  
- Quality of insights generated from the data
- User interface design and functionality
- Performance optimization for real-time processing

**Bonus Points:**

- Implement privacy-preserving techniques (face blurring, anonymous tracking)
- Add predictive analytics (peak hours, restocking predictions)
- Create mobile-responsive dashboard
- Integration with existing retail systems (inventory management)

---

## Summary

Today we've journeyed through the most sophisticated realms of computer vision, learning to build systems that can see, recognize, create, and understand visual information with remarkable sophistication. Like master chefs who can identify every ingredient by sight, track multiple dishes simultaneously, transform presentations artistically, and monitor the entire kitchen operation, you now possess the skills to create AI systems that bring superhuman visual intelligence to any domain.

These advanced computer vision techniques form the foundation for countless applications - from security systems and customer analytics to creative tools and automated monitoring. The key is understanding not just how to implement these technologies, but when and how to combine them effectively to solve real-world challenges.

Remember: the most powerful vision systems aren't just about detecting objects or recognizing faces - they're about understanding context, patterns, and meaning in visual data to make intelligent decisions and provide valuable insights.