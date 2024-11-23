 # [CISTPA 2024] FOOTBALLNET: a Deep LEARNING NETWORK for FOOTBALL COMPETITION PREDICTION

 ### [[Paper](https://github.com/Wangtongfeng1226/Wangtongfeng1226.github.io/blob/main/FOOTBALLNET-%20a%20Deep%20LEARNING%20NETWORK%20for%20FOOTBALL%20COMPETITION%20PREDICTION.pdf)] [[Project Website](https://github.com/Wangtongfeng1226/FOOTBALLNET/)]

<p align='center'>
<img src="https://github.com/Wangtongfeng1226/FOOTBALLNET/blob/main/image/WechatIMG1785.jpg" width='100%'/>
</p>

With the advancement of technology, artificial intelligence (AI) and big data have significantly impacted sports communication, particularly in enhancing audience engagement, personalized content recommendations, real-time data analysis, and targeted advertising. Predicting sports outcomes remains a challenging task due to the inherent unpredictability of factors such as player performance, weather, and tactical decisions. Traditional statistical models and machine learning algorithms, like logistic regression and XGBoost, have had limited success in capturing complex patterns in high-dimensional data. This study explores the novel application of convolutional neural networks (CNNs) for predicting sports event outcomes. Leveraging CNNs' ability to identify intricate patterns, the proposed method shows improved accuracy, precision, recall and F1 score compared to six traditional models, achieving up to 93%- 97%, 87.8%-92.2%, 91.5%-96.5% and 90.7-92.9 respectively. Besides, while some fluctuations are influenced by Gaussian noise, the four targets of our model are still much higher than other traditional models, and both could show the stability and reliability of the model. The result indicates that CNN offers a promising method for enhancing sports analytics and permits reliable and accurate predictions of complicated datasets.
<br/>

**FOOTBALLNET: a Deep LEARNING NETWORK for FOOTBALL COMPETITION PREDICTION**
<br/>
[Tongfeng Wang](https://github.com/Wangtongfeng1226/Wangtongfeng1226.github.io)
<br/>
In GUANGZHOU 2024.


 
# Prerequisite
- Insert the database
  ```
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.utils.data import DataLoader, TensorDataset
  import numpy as np
  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import roc_curve, roc_auc_score
  import matplotlib.pyplot as plt
  ```

 # Data loading and processing
 - Data loading
  ```
  rankings = pd.read_csv('../input/fifa-international-soccer-mens-ranking-1993now/fifa_ranking.csv')
  matches = pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')
  world_cup = pd.read_csv('../input/fifa-worldcup-2018-dataset/World Cup 2018 Dataset.csv')
  ```
All the data comes from three datasets: the dataset of FIFA soccer rankings from 1993 to 2018(Tadhg Fitzgerald), International football results from 1872 to 2024(Mart Jürisoo) and FIFA World Cup 2018 dataset(Nuggs).
 - Data preprocessing
  ```
  rankings = rankings.loc[:,['rank', 'country_full', 'country_abrv', 'cur_year_avg_weighted', 'rank_date', 
                           'two_year_ago_weighted', 'three_year_ago_weighted']]
  rankings = rankings.replace({"IR Iran": "Iran"})
  rankings['weighted_points'] =  rankings['cur_year_avg_weighted'] + rankings['two_year_ago_weighted'] + rankings['three_year_ago_weighted']
  rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])
  matches =  matches.replace({'Germany DR': 'Germany', 'China': 'China PR'})
  matches['date'] = pd.to_datetime(matches['date'])
  world_cup = world_cup.loc[:, ['Team', 'Group', 'First match \nagainst', 'Second match\n against', 'Third match\n against']]
  world_cup = world_cup.dropna(how='all')
  world_cup = world_cup.replace({"IRAN": "Iran", 
                               "Costarica": "Costa Rica", 
                               "Porugal": "Portugal", 
                               "Columbia": "Colombia", 
                               "Korea" : "Korea Republic"})
  world_cup = world_cup.set_index('Team')
  rankings = rankings.set_index(['rank_date'])\
            .groupby(['country_full'], group_keys=False)\
            .resample('D').first()\
            .fillna(method='ffill')\
            .reset_index()
   ```
Data replacement (country names, match date and time formats), time series padding, data merging (home and away teams)


# Data merging
- Match data and ranking data are merged
  ```
  matches = matches.merge(rankings, 
                        left_on=['date', 'home_team'], 
                        right_on=['rank_date', 'country_full'])
  matches = matches.merge(rankings, 
                        left_on=['date', 'away_team'], 
                        right_on=['rank_date', 'country_full'], 
                        suffixes=('_home', '_away'))
  ```
- Feature generation
  ```
  matches['rank_difference'] = matches['rank_home'] - matches['rank_away']  
  matches['average_rank'] = (matches['rank_home'] + matches['rank_away'])/2  
  matches['point_difference'] = matches['weighted_points_home'] - matches['weighted_points_away']  
  matches['score_difference'] = matches['home_score'] - matches['away_score'] 
  matches['is_won'] = matches['score_difference'] > 0 
  matches['is_stake'] = matches['tournament'] != 'Friendly'
  ```
In this study, six factors have been chosen as features, they are "ranking difference", "average rank", "point difference", "score difference", "is won" and "is stake". 
# Model training and prediction
- Definition of model
    ```
    class CNNModel(nn.Module):
    def __init__(self, input_channels, input_length):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64, 64)  
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            outputs = self.forward(inputs)
            return torch.cat((1 - outputs, outputs), dim=1).numpy()
        
    feature_columns = ['rank_difference', 'average_rank', 'point_difference', 'score_difference', 'is_stake']
    X = matches[feature_columns].fillna(0).values.astype(np.float32)
    y = matches['is_won'].astype(float).fillna(0).values.astype(np.float32)
    ```
    This code defines a 1D CNN model for predicting match outcomes based on features like rank difference, score difference, and point difference. The model includes three convolutional layers, max pooling, dropout, and two fully connected layers, enabling it to extract local patterns from input data. The forward method performs the forward pass, while predict_proba outputs probabilities for both winning and losing. Features and target variables are preprocessed to ensure compatibility. The model is designed for binary classification and can be optimized further by adding features or tuning hyperparameters to improve prediction accuracy.
- Data preparation
    ```
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    ```
  This code splits data into training and testing sets, converts them into PyTorch tensors with added channel dimensions, and creates a DataLoader with batch processing and shuffling for efficient model training.
  
- Model training
    ```
    def train_and_evaluate_cnn(model, train_loader, X_test, y_test, learning_rate=0.001, epochs=100):
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
     ```
    This function trains a CNN model using BCEWithLogitsLoss, the Adam optimizer, and GPU if available. It processes batches from a DataLoader, performs forward and backward passes, updates parameters, and prints epoch-wise loss for progress tracking.
- Model evaluation
    ```
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
        logits = model(X_test_tensor)
        y_pred_prob = torch.sigmoid(logits).cpu().numpy().ravel()  # 使用Sigmoid转换logits为概率
    
    y_test = np.array(y_test).ravel()
    
    unique_classes = np.unique(y_test)
    if len(unique_classes) < 2:
        print(f"Warning: Only one class present in y_test ({unique_classes}). ROC AUC score is not defined in that case.")
    else:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'CNN (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
    
     input_channels = 1  
     input_length = X_train_tensor.shape[2] 
     model_cnn = CNNModel(input_channels, input_length)

     train_and_evaluate_cnn(model_cnn, train_loader, X_test, y_test)
        
     from sklearn.linear_model import LogisticRegression
     from sklearn.tree import DecisionTreeClassifier
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.svm import SVC
     from xgboost import XGBClassifier
     from sklearn.neural_network import MLPClassifier
     from sklearn.model_selection import train_test_split
     from sklearn.metrics import roc_curve, roc_auc_score
     import matplotlib.pyplot as plt

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     models = {
         "Decision Tree": DecisionTreeClassifier(),
         "Random Forest": RandomForestClassifier(),
         "SVM": SVC(probability=True),
         "XGBoost": XGBClassifier(),
         "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000),
         "Logistic Regression": LogisticRegression(C=1e-5)
     }

     plt.figure(figsize=(10, 6))
     for name, model in models.items():
         model.fit(X_train, y_train)
         fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
         auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
         plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

     plt.plot([0, 1], [0, 1], 'k--')
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     plt.title('ROC Curve')
     plt.legend()
     plt.show() 
     ```
    This code evaluates a CNN model and compares it with traditional machine learning classifiers (e.g., Decision Tree, Random Forest, SVM, XGBoost, MLP, Logistic Regression) using ROC curves and AUC scores. The CNN evaluation involves converting test data to tensors, generating probabilities via the Sigmoid function, and plotting the ROC curve. For traditional classifiers, models are trained using scikit-learn, and their ROC curves and AUC values are computed based on test predictions. The performance of all models is visualized, highlighting their ability to distinguish between classes. 

 # Group stage simulation
 - Prepare for group stage competition
    ```
    from itertools import combinations
    margin = 0.05

    world_cup_rankings = rankings.loc[(rankings['rank_date'] == rankings['rank_date'].max()) & 
                                  rankings['country_full'].isin(world_cup.index.unique())]
    world_cup_rankings = world_cup_rankings.set_index(['country_full'])

    world_cup['points'] = 0
    world_cup['total_prob'] = 0
    ```
 - Competition simulation
    ```
    def prepare_data_for_cnn(home, away, world_cup, world_cup_rankings, model_cnn):
    home_rank = world_cup_rankings.loc[home, 'rank']
    home_points = world_cup_rankings.loc[home, 'weighted_points']
    opp_rank = world_cup_rankings.loc[away, 'rank']
    opp_points = world_cup_rankings.loc[away, 'weighted_points']

    row = pd.DataFrame({
        'rank_difference': [home_rank - opp_rank],
        'average_rank': [(home_rank + opp_rank) / 2],
        'point_difference': [home_points - opp_points],
        'score_difference': [0],
        'is_stake': [True]
    })

    row_values = row.values.astype(float).reshape(1, -1)

    
    home_win_prob = model_cnn.predict_proba(row_values)[:, 1][0]
    world_cup.loc[home, 'total_prob'] += home_win_prob
    world_cup.loc[away, 'total_prob'] += 1 - home_win_prob
    points = 0
    
    if home_win_prob <= 0.5 - margin:
        print("{} wins with {:.2f}".format(away, 1 - home_win_prob))
        world_cup.loc[away, 'points'] += 3
    elif home_win_prob > 0.5 - margin and home_win_prob < 0.5 + margin:
        print("Draw")
        world_cup.loc[home, 'points'] += 1
        world_cup.loc[away, 'points'] += 1
    elif home_win_prob >= 0.5 + margin:
        points = 3
        world_cup.loc[home, 'points'] += 3
        print("{} wins with {:.2f}".format(home, home_win_prob))

    for group in set(world_cup['Group']):
    print('___Starting group {}:___'.format(group))
    for home, away in combinations(world_cup.query('Group == "{}"'.format(group)).index, 2):
        print("{} vs. {}: ".format(home, away), end='')
        prepare_data_for_cnn(home, away, world_cup, world_cup_rankings, model_cnn)
    ```
    This code simulates all group stage matches using a CNN model to predict outcomes based on team rankings and features. It calculates points and total probabilities for each team. Matches are categorized into home win, away win, or draw based on the predicted probabilities and a defined margin. The final standings can be derived from the world_cup DataFrame after all matches are processed. This method provides a data-driven approach to simulate and analyze tournament outcomes.
 # Knockout simulation
 - Prepare for the knockout round
    ```
    pairing = [0,3,4,7,8,11,12,15,1,2,5,6,9,10,13,14]
    world_cup = world_cup.sort_values(by=['Group', 'points', 'total_prob'], ascending=False).reset_index()
    next_round_wc = world_cup.groupby('Group').nth([0, 1]) # select the top 2
    next_round_wc = next_round_wc.reset_index()
    next_round_wc = next_round_wc.loc[pairing]
    next_round_wc = next_round_wc.set_index('Team')

    finals = ['round_of_16', 'quarterfinal', 'semifinal', 'final']

    labels = list()
    odds = list()

    for f in finals: #遍历所有淘汰赛阶段。
        print("___Starting of the {}___".format(f))
        iterations = int(len(next_round_wc) / 2)
        winners = []
    ```
  - The knockout stage prediction
    ```
    for i in range(iterations):
        home = next_round_wc.index[i*2] 
        away = next_round_wc.index[i*2+1]
        print("{} vs. {}: ".format(home,away), end='')
        columns = ['average_rank', 'rank_difference', 'point_difference', 'score_difference', 'is_stake']
        row = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan, True]], columns=columns)
        
    home_rank = world_cup_rankings.loc[home, 'rank'] 
    home_points = world_cup_rankings.loc[home, 'weighted_points'] 
    opp_rank = world_cup_rankings.loc[away, 'rank'] 
    opp_points = world_cup_rankings.loc[away, 'weighted_points']
        
    row['average_rank'] = (home_rank + opp_rank) / 2 
    row['rank_difference'] = home_rank - opp_rank 
    row['point_difference'] = home_points - opp_points 
    row['score_difference'] = 0
        
    row_values = row.values.astype(float)
        
    home_win_prob = model_cnn.predict_proba(row_values)[:,1][0] 
    if home_win_prob <= 0.5:
        print("{0} wins with probability {1:.2f}".format(away, 1-home_win_prob))
        winners.append(away) 
    else:
        print("{0} wins with probability {1:.2f}".format(home, home_win_prob))
        winners.append(home)
    ```
  - Match odds and label generation
    ```
    if home_win_prob == 0:  
        home_odds = float('inf')  
    else:
        home_odds = 1 / home_win_prob  

    if home_win_prob == 1: 
        away_odds = float('inf')  
    else:
        away_odds = 1 / (1 - home_win_prob)  

    labels.append("{}({:.2f}) vs. {}({:.2f})".format(
        world_cup_rankings.loc[home, 'country_abrv'], 
        home_odds, 
        world_cup_rankings.loc[away, 'country_abrv'], 
        away_odds
    ))
    odds.append([home_win_prob, 1-home_win_prob])
                
    next_round_wc = next_round_wc.loc[winners]
    print("\n")
    ```
    This code efficiently simulates the knockout stages of a tournament using a CNN model for predictions. It calculates features for each match, predicts outcomes, determines winners, and progresses them to the next round. Match results, odds, and labels are stored for analysis or visualization, enabling both competitive insights and predictive analytics for tournament simulation.
