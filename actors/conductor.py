import random
import actors.client
import actors.server
from datetime import timedelta
from data.preprocessing_data import write_to_csv

class Conductor:
    """
    A conductor is responsible for conducting the whole learning process.

    It works like a conductor in an orchestra. It has a detailed linear plan on what to play (in our case it is the
    dataframe ordered by date) and its basic job is to choose certain members from the orchestra to start to play.

    Members:
        sheet (pd.DataFrame): the database containing processed data about the tweets
        server: an actor that plays the same role as server in the federated learning process
        K_clients (list(client)): a list of clients
    """

    def __init__(self, sheet, K_clients):
        print("conductor created")
        self.sheet = sheet
        self.server = actors.server.Server()
        self.K_clients = [actors.client.Client(k) for k in K_clients]

    def distribute_learn_sheets(self, C, day, round):
        """It distributes data between clients and make them learn on them.

        First it chooses data that has not been learnt (self.sheet.round = -1), sends it to client to learn. It also
        receives the weights from client and sends it to the servers message box. At the end of the round it asks the
        server to do its averaging on no client data.

        Arguments:
            C list(str)): list of names of participating clients
            day (datetime.date): a date, every tweets are included that has been published this day, or earlier and
        has not been processed yet.
        round (int): the number of round - it is to be written in the dataframe so that it can be analysed

        Returns:
            -
        """
        print("clients chosen: ", C)
        for c in C:
            # if self.sheet.round is -1, it means that that row in the dataframe has not been learnt yet
            c_sheet = self.sheet[(self.sheet.round == -1) & (self.sheet.renamed_user == c) &
                                 (self.sheet['date'].dt.date <= day)]   #  ~ kinda P_{k} in pseudo
                                                                        # n_{k} = len(k_sheet)
            for client in self.K_clients:
                if client.name == c:
                    #client.sheet.append(k_sheet) # this line stays if the client can learn from all their previous tweets - not in our case, still, it may prove interesting
                    client.sheet = c_sheet  # this line stays if the client only learns from the new tweets
                    w = client.learn()
                    print("Message from client", c, "goes to server.")
                    self.server.message_box.append(w)
                    break

            self.sheet.loc[(self.sheet.round == -1) & (self.sheet.renamed_user == c) &
                           (self.sheet['date'].dt.date <= day), ['round']] = round
            #if self.sheet.round is not -1, it means that that row in the dataframe has already been learnt
        print("Server calculates average weights.")
        averaged_w = self.server.average_weights()
        print("New averaged weights are sent to participating clients.")
        for client in self.K_clients:
            if client.name == c:
                client.message_box.append(averaged_w)





    def run_rounds(self):
        """
        The whole learning process happens here.

        This method goes through the data day bay day, chooses 2 clients (C=2) out of all the K_clients and invokes the
        distribute_play_sheets method on them.

        """
        K_clients = ['Monica', 'Phoebe', 'Rachel', 'Chandler', 'Joey', 'Ross']

        start_date = self.sheet['date'].min().date()
        end_date = self.sheet['date'].max().date()

        delta = end_date - start_date
        round = 1
        for day in range(delta.days + 1):
            print("\nround #", round, " -- day:", start_date + timedelta(day), "started.")
            if not K_clients:
                K_clients = ['Monica', 'Phoebe', 'Rachel', 'Chandler', 'Joey', 'Ross']
            C_clients = random.sample(K_clients, k=2)
            K_clients = [k for k in K_clients if k not in C_clients]
            self.distribute_learn_sheets(C_clients, start_date + timedelta(day), round)
            round += 1


        write_to_csv(self.sheet, './csv', '03_with_round_numbers.csv')

