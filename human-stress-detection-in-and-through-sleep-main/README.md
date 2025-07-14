# human-stress-detection-in-and-through-sleep
We developed a web application which takes the psychological data as input and predicts the stress level for the given input data.

** FOR EDUCATIONAL PURPOSE ONLY **

• THIS PROJECT IS JUST DONE TO LEARN AND UNDERSTAND MACHINE LEARNING. <br/>
• IT CAN NOT BE USED IN REAL TIME DUE TO IT'S INABILITY TO ACQUIRE THE DATA NEEDED. <br/>
• IF THE DATA CAN BE PROVIDED THROUGH EXTERNAL DEVICES, IT CAN BE USED. <br/>

In the proposed system, a framework is developed for detecting stress in sleep through the parameters - snoring range of the user, respiration rate, body temperature, limb movement rate, blood oxygen levels, eye movement, number of hours of sleep, heart rate from a dataset by using Decision Tree algorithm.
Based on the changes during sleep, stress prediction for the following day is proposed.
Stress Levels are shown as 0 – low/normal, 1 – medium low, 2 – medium, 3 – medium high, 4 – high.

The working of the system starts with the collection of data and selecting the important attributes. Then the required data is pre-processed into the required format. The data is then divided into two parts training and testing data. The algorithms are applied and the model is trained using the training data. The accuracy of the system is obtained by testing the system using the testing data. 

The final Machine Learning model is a Decision Tree Classifier which is used to learn from the dataset and predict the output classes to new unseen data. 
This model is intergrated into a website which was built using Streamlit - a framework in Python. The website is interactive and responsive.

REFERENCES: <br/>
• Dataset: Kaggle: Human Stress Detection in and through Sleep - https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep <br/>
• Article: The Effect of Psychosocial Stress on Sleep - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4266573/ <br/>
• Documentations: <br/>
  ➢ Decision Tree - https://scikit-learn.org/stable/modules/tree.html <br/>
  ➢ Streamlit - https://docs.streamlit.io/ <br/>

BIBLIOGRAPHY: <br/>
• L. Rachakonda, A. K. Bapatla, S. P. Mohanty, and E. Kougianos, “SaYoPillow: BlockchainIntegrated Privacy-Assured IoMT Framework for Stress Management Considering Sleeping Habits”, IEEE Transactions on Consumer Electronics (TCE), Vol. 67, No. 1, Feb 2021, pp. 20-29. <br/>
• L. Rachakonda, S. P. Mohanty, E. Kougianos, K. Karunakaran, and M. Ganapathiraju, “SmartPillow: An IoT based Device for Stress Detection Considering Sleeping Habits”, in Proceedings of the 4th IEEE International Symposium on Smart Electronic Systems (iSES), 2018, pp. 161--166. <br/>
• Buddi, Padmaja & Prasad, V.V. & Sunitha, K.V.N.. (2018). Machine Learning Approach for Stress Detection using Wireless Physical Activity Tracker. International Journal of Machine Learning and Computing. 8. 33-38. 10.18178/ijmlc.2018.8.1.659.
