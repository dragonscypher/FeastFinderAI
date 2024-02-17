# FeastFinderAI 🍽️

Discover the best dining spots with FeastFinderAI! This project leverages advanced spatial data analysis and machine learning to provide insights for food entrepreneurs and enthusiasts. By analyzing comprehensive restaurant data, we aim to predict top locations for new food joints, customize offerings, and forecast dining trends. 🌟

## Project Overview 📊

- **Objective**: Analyze spatial data to predict optimal locations for new food joints, tailor offerings, and enhance customer satisfaction with targeted marketing strategies. 🎯
- **Dataset**: Utilizes NYC restaurant inspection data to assess and predict restaurant ratings. The dataset can be accessed at [DOHMH New York City Restaurant Inspection Results](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j), provided by the New York City Department of Health and Mental Hygiene (2023).

## Technologies Used 💻

- **Python**: For data analysis and machine learning model development. 🐍
- **Scikit-learn & Folium**: For machine learning algorithms and map visualizations. 🌐
- **Data**: Analyzes NYC restaurant inspection data, including attributes like restaurant name (DBA), ZIPCODE, GRADE, SCORE, and geographical coordinates (Latitude, Longitude).

## Getting Started 🚀

1. Clone this repository.
2. Install required Python packages: `pandas`, `scikit-learn`, `folium`.
3. Download the dataset and update the file path in `predictive_dining.py`.
4. Run `predictive_dining.py` to start the analysis.

## Features 🔍

- Data preprocessing and analysis of restaurant spatial data.
- Predictive modeling with RandomForestClassifier.
- Recommendations for best restaurant locations based on postal code.
- Map visualization of recommended restaurants.

## Contributing 🤝

We welcome contributions! Fork the repository and submit pull requests with enhancements.

## License 📜

FeastFinderAI is open source and available under the MIT License.

---

Enjoy exploring and discovering with FeastFinderAI! 🍴
