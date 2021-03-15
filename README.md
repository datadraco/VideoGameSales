# **Video Game Sales Analysis Project**

---

This project was first developed in a jupyter notebook and then reformatted within Visual Studio Code before being imported into the GitHub repository. <br>

---

**Required libraries:**
- `pandas`
- `plotly.express`
- `scikit-learn`

---

In order to run the code correctly within VS Code, we needed to make sure that all of the above libraries had been installed into our working Anaconda environment, including `orca` for `plotly.express` which allows you to save the static images from `plotly`. Visual Studio will prompt you to install `orca` in the terminal if it has not been installed, and then no further formatting is necessary. <br>

The machine learning analysis makes use of regressor and classifier models from `scikit-learn`, all of which are imported in the top of the `vgsales.py` file.

In order to properly access the data files and save the images within VS code, the root folder for the GitHub repository must be opened within VS Code as a workspace.

---

To reproduce the analysis of this project, the only python module that must be run is `vgsales.py`, which will then generate all of the visualizations into the `images` file and print off the machine learning results into the terminal.
