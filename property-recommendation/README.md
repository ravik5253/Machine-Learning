### Targeting Strategy to Market the Right Properties.

### Task

To develop a targeting strategy to market the right properties to the right customers (accounts)

The problem is to market the right set of properties to the accounts to maximize the property sale. 
It is essential that we arrive at an optimum number of suggested properties (‘n’) that we are targeting to the customers. 
If the numbers of properties that are being suggested are high then the customer acquisition cost goes up. 
On the other hand, if the number of suggested properties is low then it is likely that we might fail to market suitable 
properties to the customer which might end up in lesser properties being converted.


### Data Set
Training Dataset

Accounts: This table has information on customers/accounts. These are the accounts of whom we are marketing the properties for sale.

Opportunities: These include the historic deals for the accounts. Basically, this gives a transaction summary of the deals that have happened between the accounts and the properties

Accounts to Properties: This table comprises information on properties that have been already bought by the accounts

Deal to Properties: This table comprises information on the deals that has materialized on the properties

Test Dataset

This dataset has the accounts for which the properties needs to be suggested. Also note that if a property has already been bought by an account, then it cannot be marketed to another account.

Properties Database

This database contains the universal list of properties and its details



#### Goals: 
Recommend the list of ‘n’ properties that should be marketed to the accounts in the Test Dataset. 
The final evaluation will include both the factors - ‘ the optimal number of properties suggested’ and ‘ 
the number of properties converted’.


**Evaluation Metric**
The test dataset contains 30 Accounts for which the participants have to recommend properties based on their likelihood to buy.

The submission will be evaluated based on F1 score, at account id level i.e. for each account id the f1 score for the recommended products is calculated. 
The final score is the average of the F1 score received for all for all accounts.


