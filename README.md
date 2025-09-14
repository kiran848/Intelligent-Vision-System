
# K-ADAA: Intelligent Vision System for Designing Personalized Apparel

**K-ADAA** is an intelligent vision system that provides accurate body measurements to enable personalized apparel design. The system leverages MediaPipe, OpenCV, and K-Means clustering to extract body dimensions from images, reducing errors compared to manual measurement methods and minimizing apparel returns in online shopping. This project is inspired by our research published in Procedia Computer Science (Volume 259, 2025, Pages 1543-1552) and focuses on improving fit accuracy, time efficiency, and sustainable apparel production.

**Project Highlights:**  
- Automated Body Measurement: Extracts key body measurements such as arm length, chest, shoulder, waist, and full height.  
- Personalized Apparel Design: Helps users select clothing sizes that match their exact body dimensions.  
- Reduced Returns: Minimizes e-commerce returns due to poor fitting.  
- Research-Backed: Implements algorithms and techniques validated in our research.

**Project Structure:**  
kiran848/
├── README.md
├── arm\_length.py
├── chest.py
├── full\_height.py
├── lower\_length.py
├── shoulder.py
├── waist.py
├── haarcascade\_frontalface\_default.xml
├── haarcascade\_fullbody.xml
├── heatmap.ipynb
├── main\_body\_measurement.ipynb
└── measurement.txt


**Installation:**  
1. Clone the repository:  
```bash
git clone https://github.com/<USERNAME>/kiran848.git
cd kiran848
````

2. Install dependencies:

```bash
pip install opencv-python mediapipe numpy matplotlib pandas scikit-learn
```

3. Run scripts or notebooks to test body measurements.

**How to Use:**

* Python Scripts: Each script calculates a specific measurement. Example:

```bash
python arm_length.py
python chest.py
```

* Notebooks: Open `main_body_measurement.ipynb` or `heatmap.ipynb` for interactive demonstrations.
* Input: Use `measurement.txt` or your own images as input for testing.

**Research Paper Reference:**
Sharma, A., Khangarot, D., Kiran, A., Sharma, A., Birla, A., "K-ADAA: Intelligent Vision System for Designing Personalized Apparel," Procedia Computer Science, Volume 259, 2025, Pages 1543-1552. [DOI: 10.1016/j.procs.2025.04.109](https://doi.org/10.1016/j.procs.2025.04.109)

This research demonstrates how combining computer vision and machine learning techniques can improve body measurement accuracy, enhance the user experience in apparel shopping, and promote sustainable fashion practices.



Do you want me to do that next?
