# SHAMSUL

**SHAMSUL\*** explores **interpretability** for **chest X-ray pathology predictions** using methods—Grad-CAM, LIME, SHAP, and LRP. It provides heatmaps and evaluation metrics for better insights into the **medical significance** of predictions made by deep learning models.

For detailed insights and methodology, please refer to the original research paper:  
[SHAMSUL: Systematic Holistic Analysis to investigate Medical Significance Utilizing Local interpretability methods in deep learning for chest radiography pathology prediction](https://journals.uio.no/NMI/article/view/10471/9743)  

###### \*"The acronym SHAMSUL, derived from a Semitic word meaning "the Sun," serves as a symbolic representation of our heatmap score-based interpretability analysis approach aimed at unveiling the medical significance inherent in the predictions of black box deep learning models."

## Demo App- [Visit https://shamsul.serve.scilifelab.se/](https://shamsul.serve.scilifelab.se/)

Explore this work using the **publicly available** demo app (**no registration needed!**):  

### Online Access
Visit [https://shamsul.serve.scilifelab.se/](https://shamsul.serve.scilifelab.se/) to use it directly.  

### OR

### Run Locally 

#### Step 1: Install Docker  
Install Docker Engine or Docker Desktop on your system by following the [official Docker installation guide](https://docs.docker.com/get-docker/).  

#### Step 2: Launch the App  
1. Open a Terminal (or Windows Terminal).  
2. Run this command to download and start the app:  
   ```bash
   docker run --rm -p 7860:7860 mahbub1969/shamsul:v2
   ```  
3. Open your browser and go to [http://localhost:7860/](http://localhost:7860/) to use the app.  

#### Step 3: Stop the App  
To stop the app, press **Control+C** in the terminal. Note that the session won’t be saved, so the app will reset to its default state the next time you run it.  

#### Step 4: Remove the Docker Image (Optional)  
If you want to free up space, you can remove the Docker image. Use this command in your terminal:  
```bash
docker image rm mahbub1969/shamsul:v2
```  
For more details, check out the [Docker image removal guide](https://docs.docker.com/reference/cli/docker/image/rm/).

### [SHAMSUL Demo at a glance - Visit https://shamsul.serve.scilifelab.se/](https://shamsul.serve.scilifelab.se/)

![Shamsul Demo at a glance](https://raw.githubusercontent.com/anondo1969/SHAMSUL/refs/heads/main/SHAMSUL_demo_app_screnshot.png)

## Key Features

*   **Multi-Method Interpretability**: Incorporates four advanced interpretability methods—LIME, SHAP, Grad-CAM, and LRP—to provide diverse insights into deep learning model predictions.
    
*   **Focus on Medical Significancee**: Designed specifically for chest radiography pathology prediction, ensuring results are meaningful for clinical applications.
    
*   **Comprehensive Visualizations**: Generates heatmaps and segmentations to help identify the regions of interest linked to specific pathologies.
    
*   **Multi-Label, Multi-Class Analysis**: Supports analyzing both single-label and multi-label instances, accommodating a variety of medical imaging needs.
    
*   **Quantitative and Qualitative Evaluation**: Offers metrics like Intersection over Union (IoU) and detailed visual comparisons with expert annotations for robust performance assessment.
    
*   **Integration with CheXpert Dataset**: One of the largest chest X-ray datasets to validate predictions and ensure high-quality results.
    
*   **User-Friendly Interface**: Simplifies interaction by allowing users to upload images.
    
*   **Open-Source Access**: Code and resources are available, promoting transparency and enabling further development by the research community.


### An excerpt of the [SHAMSUL paper](https://doi.org/10.5617/nmi.10471)


![An excerpt of the paper](https://raw.githubusercontent.com/anondo1969/SHAMSUL/main/codes/excerpt.png)

## Citation

Please acknowledge the following work in papers or derivative software:

M. U. Alam, J. Hollmén, J. R. Baldvinsson, and R. Rahmani, “SHAMSUL: Systematic Holistic Analysis to investigate Medical Significance Utilizing
Local interpretability methods in deep learning for chest radiography pathology prediction,” Nordic Machine Intelligence, vol. 3, pp. 27–47, 2023. [https://doi.org/10.5617/nmi.10471](https://doi.org/10.5617/nmi.10471)

### Bibtex Format for Citation

```
@article{alam2023shamsul,
  title={SHAMSUL: Systematic Holistic Analysis to investigate Medical Significance Utilizing Local interpretability methods in deep learning for chest radiography pathology prediction},
  author={Ul Alam, Mahbub and Hollmén, Jaakko and Baldvinsson, Jón Rúnar and Rahmani, Rahim},
  journal={Nordic Machine Intelligence},
  volume={3},
  number={1},
  pages={27--47},
  year={2023},
  doi={10.5617/nmi.10471}
}
```
