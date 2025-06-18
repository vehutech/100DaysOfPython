### **Day 33: Images and Multimedia — Garnishing Your Plate** 🎨  

Welcome back, chefs! Yesterday, you organized your **pantry** with lists and tables. Today, we’re adding the **garnishes**—images, audio, and video—that make your web pages visually delicious.  

By the end of today, you’ll:  
✔ **Embed images** (`<img>`) with proper sizing and accessibility.  
✔ **Add videos** (`<video>`) and audio (`<audio>`) for rich media.  
✔ **Optimize files** for fast loading.  

Let’s make your dishes Instagram-worthy!  

---

## **📸 1. Images: The Visual Garnish**  

### **Basic Image Tag (`<img>`)**  
```html
<img src="pizza.jpg" alt="Homemade Margherita Pizza" width="400">
```
- **`src`**: Image path (URL or local file).  
- **`alt`**: Description for accessibility (screen readers).  
- **`width`/`height`**: Control display size (use CSS for responsiveness).  

### **Image Best Practices**  
✅ **Use web-friendly formats**:  
- `.jpg` (photos)  
- `.png` (transparency)  
- `.webp` (modern, smaller files)  

✅ **Optimize size**: Compress images with [TinyPNG](https://tinypng.com/).  

🚫 **Avoid giant files** (slow loading).  

---

## **🎥 2. Video: Adding Motion**  

### **Embedding Videos (`<video>`)**  
```html
<video controls width="500">
    <source src="pizza-tutorial.mp4" type="video/mp4">
    Your browser doesn’t support videos.
</video>
```
- **`controls`**: Adds play/pause buttons.  
- **`<source>`**: Provides multiple formats (e.g., `.mp4`, `.webm`).  

### **YouTube Embed (Bonus)**  
```html
<iframe width="500" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>
```

---

## **🎵 3. Audio: Background Flavors**  

### **Adding Sound (`<audio>`)**  
```html
<audio controls>
    <source src="kitchen-sounds.mp3" type="audio/mpeg">
    Your browser doesn’t support audio.
</audio>
```
- Great for podcasts, music, or ambiance.  

---

## **🍽️ 4. Assignment: Build a Multimedia Recipe Page**  

**📝 Task:**  
1. Create a `multimedia-recipe.html` file.  
2. Add:  
   - A hero image of your dish.  
   - A cooking video (or placeholder).  
   - A background audio clip (optional).  
3. **Bonus**: Use `<figure>` and `<figcaption>` for image captions.  

**Example:**  
```html
<!DOCTYPE html>
<html>
<head>
    <title>Multimedia Pizza Recipe</title>
</head>
<body>
    <h1>🍕 Video Pizza Tutorial</h1>
    <img src="pizza.jpg" alt="Finished Pizza" width="600">
    <figure>
        <video controls width="600">
            <source src="pizza-tutorial.mp4" type="video/mp4">
        </video>
        <figcaption>Watch the step-by-step guide.</figcaption>
    </figure>
</body>
</html>
```

---

## **📚 Resources**  
- [MDN Images Guide](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/img)  
- [HTML Video Tutorial](https://www.w3schools.com/html/html5_video.asp)  

**Tomorrow:** We’ll take orders with **HTML forms**! 📝