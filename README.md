# capstoneproject

[Dynamic Pricing for Urban Parking Lots](./Dynamic_Pricing_for_Urban_ParkingLots.pdf)

---

## 🔧 Tech Stack

| Category                 | Tools / Libraries Used                                                               |
|--------------------------|--------------------------------------------------------------------------------------|
| **Programming Language** | Python 3                                                                             |
| **Data Manipulation**    | NumPy, Pandas                                                                        |
| **Visualization**        | Bokeh, Matplotlib, Panel, IPython.display                                            |
| **Streaming & Pipelines**| Pathway (real-time data streaming and transformation)                                |
| **Geospatial Analysis**  | scikit-learn (BallTree), geopy                                                       |
| **Notebook Environment** | Google Colab                                                                         |
| **Version Control**      | Git, GitHub                                                                          |
| **Miscellaneous**        | `time.sleep`, `random`, `datetime`                                                  |

---

## 🏗️ Architecture Diagram

```mermaid
graph TD
    A["Raw Input Data - CSV or Stream"] --> B["Data Cleaning"]
    B --> C["Feature Engineering"]
    C --> D1["Model 1 - Rule-Based Pricing"]
    C --> D2["Model 2 - Demand-Based Pricing"]
    C --> D3["Model 3 - Competition-Aware Pricing"]
    D1 --> E["Price Output"]
    D2 --> E
    D3 --> E
    E --> F["Real-Time Visualization with Bokeh"]
    F --> G["User Dashboard or Monitoring Panel"]
