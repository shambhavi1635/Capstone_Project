# capstoneproject

[Dynamic Pricing for Urban Parking Lots](./Dynamic Pricing for Urban Parking Lots.pdf)

## 🔧 Tech Stack

| Category               | Tools / Libraries Used                                                                 |
|------------------------|----------------------------------------------------------------------------------------|
| **Programming Language** | Python 3                                                                              |
| **Data Manipulation**     | NumPy, Pandas                                                                        |
| **Visualization**         | Bokeh (interactive plots), Matplotlib, Panel (layout), IPython.display              |
| **Streaming & Pipelines** | Pathway (real-time data streaming and transformation)                                |
| **Geospatial Analysis**   | scikit-learn (BallTree for proximity), geopy (if used elsewhere for coordinates)     |
| **Notebook Environment**  | Google Colab (development and execution)                                             |
| **Version Control**       | Git, GitHub                                                                          |
| **Miscellaneous**         | `time.sleep`, `random`, `datetime` (for simulation and time-based operations)        |




## 🏗️ Architecture Diagram

```mermaid
graph TD
    A["Raw Input Data - CSV or Stream"] --> B["Data Cleaning and Interpolation"]
    B --> C["Feature Engineering"]
    C --> D1["Model 1 - Rule-Based Pricing"]
    C --> D2["Model 2 - Demand-Based Pricing"]
    C --> D3["Model 3 - Competition-Aware Pricing"]
    D1 --> E["Price Output"]
    D2 --> E
    D3 --> E
    E --> F["Real-Time Visualization with Bokeh"]
    F --> G["User Dashboard or Monitoring Panel"]






## 📄 Final Report

You can view the complete project report here:

[👉 Click to view the Report (PDF)](./docs/Dynamic Pricing for Urban Parking Lots.pdf)














 
