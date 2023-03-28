import numpy as np
import pandas as pd
from datetime import datetime as dt
import re
import time
from sdv.tabular import GaussianCopula, CTGAN, CopulaGAN, TVAE
from sdv.constraints import Inequality, create_custom_constraint
from copy import deepcopy
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")

def is_valid_arithmetic_equality(column_names, data):
    is_equal = [data[column_names[0]] - data[column_names[1]] == data[column_names[2]]][0]
    return is_equal

arithmetic_equality_constraint = create_custom_constraint(is_valid_fn = is_valid_arithmetic_equality)

def is_valid_arithmetic_inequality(column_names, data):
    is_equal = [data[column_names[0]] >= data[column_names[1]] + column_names[2]][0]
    return is_equal

arithmetic_inequality_constraint = create_custom_constraint(is_valid_fn = is_valid_arithmetic_inequality)

def is_valid_inclusive(column_names, data):
    is_equal = []
    for index, row in data.iterrows():
        is_equal.append(str(int(row[column_names[0]]))[column_names[2]:column_names[2]+len(str(int(row[column_names[1]])))] == str(int(row[column_names[1]])))
    is_equal = pd.Series(i for i in is_equal)
    return is_equal

inclusive_constraint = create_custom_constraint(is_valid_fn = is_valid_inclusive)

class SD_generator():
    """
    A class to detect deterministic relationships between two/three columns from a given dataset.
    Create constraints with detected relationships.
    Apply models in SDV and generate synthetic data.

    ...

    Attributes
    ----------
    data : Pandas DataFrame
        an input dataset in Pandas DataFrame format

    inequality_threshold : float
        a cut-off percentage for detection functions to confirm the inequality deterministic relationships

    arithmetic_equality_threshold : float
        a cut-off percentage for detection functions to confirm the arithmetic equality deterministic relationships

    inclusive_threshold : float
        a cut-off percentage for detection functions to confirm the inclusive deterministic relationships

    inequality_dict : dictionary
        stores inequality deterministic relationships;
        in which the key is greater than its values

    inequality_runtime : float
        a variable to store the runtime of inequality detection function

    arithmetic_equality_dict : dictionary
        stores deterministic relationships like "A = B + C" among three colomns

    arithmetic_equality_runtime : float
        a variable to store the runtime of arithmetic equality detection function

    arithmetic_equality_flag : bool
        a bolean variable; if True, apply arithmetic equality constraints to generate synthetic data

    arithmetic_inequality_dict : dictionary
        stores deterministic relationships like "A >= B + X" between two colomns

    arithmetic_inequality_runtime : float
        a variable to store the runtime of arithmetic inequality detection function

    inclusive_dict : dictionary
        stores relationships that a column contains another column and the starting index

    inclusive_runtime : float
        a variable to store the runtime of inclusive detection function

    inclusive_flag : bool
        a bolean variable; if True, apply inclusive constraints to generate synthetic data

    constraints : list
        a list containing constraints for synthetic data generation model training

    models : dictionary
        an empty dictionary to store SDV models that are ready to generate synthetic data

    temp_dict : dictionary
        an empty dictionary to store temporary relationships

    Methods
    -------
    preprocess():
        Change the dtpyes of date columns to float and
        drop the rows of the input dataframe which have missing values.

    detect_inequality():
        Detect the inequality deterministic relationship between two colomns.

    detect_arithmetic_equality():
        Detect the deterministic relationships like "A = B + C" among three colomns.

    detect_arithmetic_inequality():
        Detect the deterministic relationships like "A >= B + X" between two colomns.

    detect_inclusive():
        Detect the inclusive relationships that a column contains another column and the starting index.

    create_constraints():
        Create constraints for synthetic data generation model training.

    apply_model():
        Train a specific model in SDV with constraints.
        Store the trained model in dictionary "models".

    generate():
        Generate synthetic data of a specific number of rows with a specific pre-trainned model.
    """
    def __init__(self, data, inequality_threshold, arithmetic_equality_threshold, inclusive_threshold,
                 inequality_dict=None, inequality_runtime=0,
                 arithmetic_equality_dict=None, arithmetic_equality_runtime=0, arithmetic_equality_flag=False,
                 arithmetic_inequality_dict=None, arithmetic_inequality_runtime=0,
                 inclusive_dict=None, inclusive_runtime=0, inclusive_flag=False,
                 constraints=None, models=None, temp_dict=None):
        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
            dataframe : Pandas DataFrame
                an input dataset in Pandas DataFrame format

            threshold : float
                a cut-off percentage for detection functions to confirm the deterministic relationships

            inequality_dict : dictionary
                an empty dictionary to store inequality deterministic relationships

            inequality_runtime : float
                a variable to store the runtime of inequality detection function

            arithmetic_equality_dict : dictionary
                an empty dictionary to store deterministic relationships among three columns like "A = B + C"

            arithmetic_equality_runtime : float
                a variable to store the runtime of arithmetic equality detection function

            arithmetic_equality_flag : bool
                a bolean variable; if True, apply arithmetic equality constraints to generate synthetic data

            arithmetic_inequality : dictionary
                an empty dictionary to store deterministic relationships between two columnslike "A >= B + X"

            arithmetic_inequality_runtime : float
                a variable to store the runtime of arithmetic inequality detection function

            inclusive_flag : bool
                a bolean variable; if True, apply inclusive constraints to generate synthetic data

            constarints : list
                an empty list to store contraints for synthetic data generation model training

            models : dictionary
                an empty dictionary to store SDV models that are ready to generate synthetic data

            temp_dict : dictionary
                an empty dictionary to store temporary relationships
        """
        self.data = data
        self.inequality_threshold = inequality_threshold
        self.arithmetic_equality_threshold = arithmetic_equality_threshold
        self.inclusive_threshold = inclusive_threshold
        self.inequality_dict = inequality_dict if inequality_dict is not None else {}
        self.inequality_runtime = inequality_runtime
        self.arithmetic_equality_dict = arithmetic_equality_dict if arithmetic_equality_dict is not None else {}
        self.arithmetic_equality_runtime = arithmetic_equality_runtime
        self.arithmetic_equality_flag = arithmetic_equality_flag
        self.arithmetic_inequality_dict = arithmetic_inequality_dict if arithmetic_inequality_dict is not None else {}
        self.arithmetic_inequality_runtime = arithmetic_inequality_runtime
        self.inclusive_dict = inclusive_dict if inclusive_dict is not None else {}
        self.inclusive_runtime = inclusive_runtime
        self.inclusive_flag = inclusive_flag
        self.constraints = constraints if constraints is not None else []
        self.models = models if models is not None else {}
        self.temp_dict = temp_dict if temp_dict is not None else {}


    def preprocess(self):
        """
        Change date columns to float format;
        Handle missing values of the input dataframe;
        Drop the rows with missing values.

        Returns:
            None.

        Output:
            Running finished message with execution time.
        """
        st = time.time()

        ref_dt = pd.Timestamp('1900-01-01')

        str2date = lambda x: dt.strptime(x, "%Y-%m-%d") - ref_dt if x.replace(" ", "") else np.nan

        for col in self.data.columns:

            try:
                re.match('^[0-9]{4}\-[0-9]{2}\-[0-9]{2}$', self.data[col][0])

                self.data[col] = self.data[col].apply(str2date)
                self.data[col] = (self.data[col] / np.timedelta64(1, 'D')).astype(float)

            except:
                pass

        self.data.dropna(axis=0, inplace=True)

        et = time.time()
        elapsed_time = et - st
        print("Date types reformatted and missing values handled successfully!\nExecution Time:"
              , round(elapsed_time, 4), "seconds")

    def detect_inequality(self):
        """
        Detect the inequality deterministic relationship between colomns;
        Update the inequality_dictionary of the class object.

        Returns:
            None.

        Output:
            Number of relationships detected with execution time.
        """
        # Check if the inequality_dict is empty
        # If not, skip the detetcion to avoid adding duplicate records
        if bool(self.inequality_dict):
            return

        st = time.time()

        # Looping through all pair combinitions of columns
        # For each pair, check the inequality row by row
        # Compute the inequality ratio and compared it to the predefined threshold
        # If the ratio is greater than the threshold, the inequality is confirmed
        column_pairs = list(combinations(self.data.columns, 2))
        for column_pair in column_pairs:
            if self.data[column_pair[0]].dtypes in ['int', 'float'] and self.data[column_pair[1]].dtypes in ['int', 'float']:
                temp = self.data[[column_pair[0], column_pair[1]]].apply(lambda x: x[column_pair[0]] > x[column_pair[1]], axis=1)
                count_true = temp[temp == True].count()
                ratio = float(count_true) / len(temp)

                if ratio >= self.inequality_threshold:
                    if column_pair[0] in self.inequality_dict.keys():
                        self.inequality_dict[column_pair[0]].append(column_pair[1])
                    else:
                        self.inequality_dict[column_pair[0]] = []
                        self.inequality_dict[column_pair[0]].append(column_pair[1])

                elif (1 - ratio) >= self.inequality_threshold:
                    if column_pair[1] in self.inequality_dict.keys():
                        self.inequality_dict[column_pair[1]].append(column_pair[0])
                    else:
                        self.inequality_dict[column_pair[1]] = []
                        self.inequality_dict[column_pair[1]].append(column_pair[0])

        # Store the full relationships in a temp dictionary for later use
        self.temp_dict = deepcopy(self.inequality_dict)

        # Merge duplicates records and remove redundant relationships
        # Loop through the keys and their values in the inequality dictionary
        # Check if a value A is a value of value B, and value B is a key of the inequality dictionary
        # If True, change the value A to be 'N/A', and removed all 'N/A' after the looping is finished
        for key in self.inequality_dict:
            for i in range(len(self.inequality_dict[key])):
                str_1 = self.inequality_dict[key][i]
                for j in range(i, len(self.inequality_dict[key])):
                    str_2 = self.inequality_dict[key][j]
                    if ((str_1 in self.inequality_dict) and (str_2 in self.inequality_dict[str_1])):
                        self.inequality_dict[key][j] = 'N/A'
                    elif ((str_2 in self.inequality_dict) and (str_1 in self.inequality_dict[str_2])):
                        self.inequality_dict[key][i] = 'N/A'
        for key in self.inequality_dict:
            self.inequality_dict[key] = [i for i in self.inequality_dict[key] if i != 'N/A']

        et = time.time()
        self.inequality_runtime = et - st

        num = 0
        for key in self.inequality_dict:
            num += len(self.inequality_dict[key])
        print(num, "relationships detected")
        print("Execution Time:", round(self.inequality_runtime, 4), "seconds")

    def detect_arithmetic_equality(self):
        """
        Detect the deterministic relationships "A = B + C" among three columns;
        Based on the dictionary of inequality deterministic relationships.

        Returns:
            None.

        Output:
            Number of relationships detected with execution time.
        """
        # Check if the arithmetic_equality_dict is empty
        # If not, skip the detetcion to avoid adding duplicate records
        if bool(self.arithmetic_equality_dict):
            return

        st = time.time()

        # Loop through the keys of the full inequality relationships dictionary
        # Loop through the value pairs within the same key
        # Check if the percentage of row that key = sum(value_pairs) is >= the predefined threshold
        # If True, append the arithmetic equality relationship
        for key in self.temp_dict:
            column_pairs = list(combinations(self.temp_dict[key], 2))
            for column_pair in column_pairs:
                temp = self.data[[key, column_pair[0], column_pair[1]]].apply(lambda x: x[key] == x[column_pair[0]] + x[column_pair[1]], axis=1)
                count_true = temp[temp == True].count()
                if float(count_true) / len(temp) >= self.arithmetic_equality_threshold:
                    if key in self.arithmetic_equality_dict.keys():
                        self.arithmetic_equality_dict[key].append([column_pair[0], column_pair[1]])
                    else:
                        self.arithmetic_equality_dict[key] = []
                        self.arithmetic_equality_dict[key].append([column_pair[0], column_pair[1]])

        et = time.time()
        self.arithmetic_equality_runtime = et - st

        num = 0
        for key in self.arithmetic_equality_dict:
            num += len(self.arithmetic_equality_dict[key])
        print(num, "relationships detected")
        print("Execution Time:", round(self.arithmetic_equality_runtime, 4), "seconds")

    def detect_arithmetic_inequality(self):
        """
        Detect the deterministic relationships "A >= B + X" between two columns;
        Based on the dictionary of inequality deterministic relationships.

        Returns:
            None.

        Output:
            Number of relationships detected with execution time.
        """
        # Check if the arithmetic_inequality_dict is empty
        # If not, skip the detetcion to avoid adding duplicate records
        if bool(self.arithmetic_inequality_dict):
            return

        st = time.time()

        # Loop through all keys in the inequality dictionary
        # Loop through all values within the same key
        # Find the minimum difference between the key and its values
        # Append the arithmetic inequality relationship
        for key in self.inequality_dict:
            for value in self.inequality_dict[key]:
                diff = []
                for index, row in self.data.iterrows():
                    diff.append(row[key] - row[value])

                if key in self.arithmetic_inequality_dict.keys():
                    self.arithmetic_inequality_dict[key].append([value, min(diff)])
                else:
                    self.arithmetic_inequality_dict[key] = []
                    self.arithmetic_inequality_dict[key].append([value, min(diff)])

        et = time.time()
        self.arithmetic_inequality_runtime = et - st

        num = 0
        for key in self.arithmetic_inequality_dict:
            num += len(self.arithmetic_inequality_dict[key])
        print(num, "relationships detected")
        print("Execution Time:", round(self.arithmetic_inequality_runtime, 4), "seconds")

    def detect_inclusive(self):
        """
        Detect the inclusive relationships between two columns

        Returns:
            None.

        Output:
            Number of relationships detected with execution time.
        """
        # Check if the inclusive_dict is empty
        # If not, skip the detetcion to avoid adding duplicate records
        if bool(self.inclusive_dict):
            return

        st = time.time()

        data_length = len(self.data.index)

        column_pairs = list(combinations(self.data.columns, 2))

        # Loop through all column pairs in the dataset
        # Check if the percentage of the data within column A that is part of column B is >= the predefined threshold
        # If True, append the inclusive relationship
        for column_pair in column_pairs:
            ratio = 0

            if len(str(int(self.data.iloc[0][column_pair[1]]))) > len(str(int(self.data.iloc[0][column_pair[0]]))):
                temp = self.data[[column_pair[0], column_pair[1]]].apply(lambda x: str(x[column_pair[0]]) in str(x[column_pair[1]]), axis=1)
                count = temp[temp == True].count()
                ratio = float(count) / len(temp)
                key = column_pair[1]
                value = column_pair[0]
                index = str(self.data.loc[0][column_pair[1]]).find(str(self.data.loc[0][column_pair[0]]))

            elif len(str(int(self.data.iloc[0][column_pair[0]]))) > len(str(int(self.data.iloc[0][column_pair[1]]))):
                temp = self.data[[column_pair[0], column_pair[1]]].apply(lambda x: str(x[column_pair[1]]) in str(x[column_pair[0]]), axis=1)
                count = temp[temp == True].count()
                ratio = float(count) / len(temp)
                key = column_pair[0]
                value = column_pair[1]
                index = str(int(self.data.loc[0][column_pair[0]])).find(str(int(self.data.loc[0][column_pair[1]])))

            if ratio >= self.inclusive_threshold:

                if key in self.inclusive_dict.keys():
                        self.inclusive_dict[key].append([value, index])
                else:
                    self.inclusive_dict[key] = []
                    self.inclusive_dict[key].append([value, index])

        et = time.time()
        self.inclusive_runtime = et - st

        num = 0
        for key in self.inclusive_dict:
            num += len(self.inclusive_dict[key])
        print(num, "relationships detected")
        print("Execution Time:", round(self.inclusive_runtime, 4), "seconds")

    def create_constraints(self, inequality=False,
                           arithmetic_equality=False,
                           arithmetic_inequality=False,
                           inclusive=False):
        """
        Create constraints for synthetic data generation model training.

        Parameters:
            inequality: bool, default=False
                If True, create constraints for inequality deterministic relationships.

            arithmetic_equality: bool, default=False
                If True, create constraints for deterministic relationships like "A = B + C".

            arithmetic_inequality: bool, default=False
                If True, create constraints for deterministic relationships like "A >= B + X".

            inclusive: bool, default=False
                If True, create constraints for inclusive relationships.

        Returns:
            None.

        Output:
            Running finished message with execution time.
        """
        st = time.time()

        if inequality:
            for key in self.inequality_dict:
                for value in self.inequality_dict[key]:
                    self.constraints.append(Inequality(low_column_name=value, high_column_name=key))

        if arithmetic_equality:
            self.arithmetic_equality_flag=True

        if arithmetic_inequality:
            for key in self.arithmetic_inequality_dict:
                for value_list in self.arithmetic_inequality_dict[key]:
                    columns = [key, value_list[0], value_list[1]]
                    cons = arithmetic_inequality_constraint(column_names=columns)
                    self.constraints.append(cons)

        if inclusive:
            self.inclusive_flag=True

        et = time.time()
        elapsed_time = et - st

        print("Constrainsts created successfully!\nExecution Time:"
              , round(elapsed_time, 4), "seconds")

    def apply_model(self, model_name=None):
        """
        Train a specific model in SDV with constraints.
        Store the trained model in dictionary "models".

        Parameters:
            model_name: string
                The name of the models in SDV.
                Eg. "GaussianCopula", "CTGAN", "CopulaGAN" and "TVAE".

        Returns:
            None.
        """
        st = time.time()

        # Check if the input of model name is missing
        if model_name == None:
            print("No input for model name!")
            return

        elif model_name == 'GaussianCopula':
            model = GaussianCopula(constraints=self.constraints)
            model.fit(self.data)
            self.models['GaussianCopula'] = model

            et = time.time()
            elapsed_time = et - st

            print(f"Execution Time for training {model_name}:", round(elapsed_time, 4), "seconds")


        elif model_name == 'CTGAN':
            model = CTGAN(constraints=self.constraints)
            model.fit(self.data)
            self.models['CTGAN'] = model

            et = time.time()
            elapsed_time = et - st

            print(f"Execution Time for training {model_name}:", round(elapsed_time, 4), "seconds")

        elif model_name == 'CopulaGAN':
            model = CopulaGAN(constraints=self.constraints)
            model.fit(self.data)
            self.models['CopulaGAN'] = model

            et = time.time()
            elapsed_time = et - st

            print(f"Execution Time for training {model_name}:", round(elapsed_time, 4), "seconds")

        elif model_name == 'TVAE':
            model = TVAE(constraints=self.constraints)
            model.fit(self.data)
            self.models['TVAE'] = model

            et = time.time()
            elapsed_time = et - st

            print(f"Execution Time for training {model_name}:", round(elapsed_time, 4), "seconds")

        # Handle input of wrong model name
        else:
            print("Wrong model name!")

    def generate(self, model_name=None, num_rows=0):
        """
        Generate synthetic data of a specific number of rows with a specific pre-trainned model.

        Parameters:
            model_name: string
                The name of the models in SDV.
                Eg. "GaussianCopula", "CTGAN", "CopulaGAN" and "TVAE".

            num_rows: integer
                Number of rows needed to generate.

        Returns:
            Synthetic data in DataFrame type.
        """
        st = time.time()

        # Handle wrong/missing input of model name or number of rows for generation
        if model_name == None:
            print("No input for model name!")
            return None
        elif (num_rows == 0) or (type(num_rows) != int):
            print("Number of rows has to be integer and greater than 0!")
            return None

        elif model_name not in ["GaussianCopula", "CTGAN", "CopulaGAN", "TVAE"]:
            print("Wrong model name!\nAccepted model name: GaussianCopula, CTAGAN, CopulaGAN and TVAE.")
            return None

        else:
            syn_data = self.models[model_name].sample(num_rows=num_rows)

            if self.arithmetic_equality_flag:
                for key in self.arithmetic_equality_dict:
                    syn_data[key] = syn_data[self.arithmetic_equality_dict[key][0][0]] + syn_data[self.arithmetic_equality_dict[key][0][1]]

            if self.inclusive_flag:
                for key in self.inclusive_dict:
                    for value_list in self.inclusive_dict[key]:
                        for index, row in syn_data.iterrows():
                            list_temp = list(str(int(row[key])))
                            list_temp[value_list[1] : value_list[1] + len(str(row[value_list[1]]))] = str(int(row[value_list[0]]))
                            syn_data.at[index, key] = int(''.join(list_temp))

            et = time.time()
            elapsed_time = et - st
            print(f"Synthetic data generated successfully with {model_name} model!\nExecution Time:", round(elapsed_time, 4), "seconds")

            return syn_data
