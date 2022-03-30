import 'package:flutter/material.dart';
import 'package:mmr_app/Widgets/main_card.dart';

class MoviePage extends StatelessWidget {
  const MoviePage({Key? key}) : super(key: key);
  static const List<String> movie = [
    "Toy Story (1995)",
    "Jumanji (1995)",
    "Grumpier Old Men (1995)",
    "Waiting to Exhale (1995)",
    "Father of the Bride Part II (1995)",
    "Heat (1995)",
    "Sabrina (1995)",
    "Tom and Huck (1995)",
    "Sudden Death (1995)",
    "GoldenEye (1995)",
    "American President",
    "Dracula: Dead and Loving It (1995)",
    "Balto (1995)",
    "Nixon (1995)",
    "Cutthroat Island (1995)",
    "Casino (1995)",
    "Sense and Sensibility (1995)",
    "Four Rooms (1995)",
    "Ace Ventura: When Nature Calls (1995)",
    "Money Train (1995)",
    "Get Shorty (1995)",
    "Copycat (1995)",
    "Assassins (1995)",
    "Powder (1995)",
    "Leaving Las Vegas (1995)",
    "Othello (1995)",
    "Now and Then (1995)",
    "Persuasion (1995)",
    "City of Lost Children",
    "Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)",
    "Dangerous Minds (1995)",
    "Twelve Monkeys (a.k.a. 12 Monkeys) (1995)",
    "Babe (1995)",
    "Dead Man Walking (1995)",
    "It Takes Two (1995)",
    "Clueless (1995)",
    "Cry",
    "Richard III (1995)",
    "Dead Presidents (1995)",
    "Restoration (1995)",
    "Mortal Kombat (1995)",
    "To Die For (1995)",
    "How to Make an American Quilt (1995)",
    "Seven (a.k.a. Se7en) (1995)",
    "Pocahontas (1995)",
    "When Night Is Falling (1995)",
    "Usual Suspects",
    "Mighty Aphrodite (1995)",
    "Lamerica (1994)",
    "Big Green",
  ];
  static const List<String> genre = [
    "Adventure|Animation|Children|Comedy|Fantasy",
    "Adventure|Children|Fantasy",
    "Comedy|Romance",
    "Comedy|Drama|Romance",
    "Comedy",
    "Action|Crime|Thriller",
    "Comedy|Romance",
    "Adventure|Children",
    "Action",
    "Action|Adventure|Thriller",
    "Comedy|Horror",
    "Adventure|Animation|Children",
    "Drama",
    "Action|Adventure|Romance",
    "Crime|Drama",
    "Drama|Romance",
    "Comedy",
    "Comedy",
    "Action|Comedy|Crime|Drama|Thriller",
    "Comedy|Crime|Thriller",
    "Crime|Drama|Horror|Mystery|Thriller",
    "Action|Crime|Thriller",
    "Drama|Sci-Fi",
    "Drama|Romance",
    "Drama",
    "Children|Drama",
    "Drama|Romance",
    " The (Cité des enfants perdus",
    "Crime|Drama",
    "Drama",
    "Mystery|Sci-Fi|Thriller",
    "Children|Drama",
    "Crime|Drama",
    "Children|Comedy",
    "Comedy|Romance",
    "Drama|War",
    "Action|Crime|Drama",
    "Drama",
    "Action|Adventure|Fantasy",
    "Comedy|Drama|Thriller",
    "Drama|Romance",
    "Mystery|Thriller",
    "Animation|Children|Drama|Musical|Romance",
    "Drama|Romance",
    "Comedy|Drama|Romance",
    "Adventure|Drama",
  ];
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Movies'),
        backgroundColor: Colors.red,
        actions: [
          IconButton(
              onPressed: () {
                showSearch(
                  context: context,
                  delegate: CustomSearchDelegate(),
                );
              },
              icon: const Icon(Icons.search))
        ],
      ),
      body: Container(
        child: ListView.builder(itemCount: genre.length, itemBuilder: (context, index) {
          final name = movie[index];
          final gen = genre[index];
          return ItemCard(name, '4.5', gen, context);
        }),
      ),
    );
  }
}

class CustomSearchDelegate extends SearchDelegate {
  List<String> searchTerm = [
    "Toy Story (1995)",
    "Jumanji (1995)",
    "Grumpier Old Men (1995)",
    "Waiting to Exhale (1995)",
    "Father of the Bride Part II (1995)",
    "Heat (1995)",
    "Sabrina (1995)",
    "Tom and Huck (1995)",
    "Sudden Death (1995)",
    "GoldenEye (1995)",
    "American President",
    "Dracula: Dead and Loving It (1995)",
    "Balto (1995)",
    "Nixon (1995)",
    "Cutthroat Island (1995)",
    "Casino (1995)",
    "Sense and Sensibility (1995)",
    "Four Rooms (1995)",
    "Ace Ventura: When Nature Calls (1995)",
    "Money Train (1995)",
    "Get Shorty (1995)",
    "Copycat (1995)",
    "Assassins (1995)",
    "Powder (1995)",
    "Leaving Las Vegas (1995)",
    "Othello (1995)",
    "Now and Then (1995)",
    "Persuasion (1995)",
    "City of Lost Children",
    "Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)",
    "Dangerous Minds (1995)",
    "Twelve Monkeys (a.k.a. 12 Monkeys) (1995)",
    "Babe (1995)",
    "Dead Man Walking (1995)",
    "It Takes Two (1995)",
    "Clueless (1995)",
    "Cry",
    "Richard III (1995)",
    "Dead Presidents (1995)",
    "Restoration (1995)",
    "Mortal Kombat (1995)",
    "To Die For (1995)",
    "How to Make an American Quilt (1995)",
    "Seven (a.k.a. Se7en) (1995)",
    "Pocahontas (1995)",
    "When Night Is Falling (1995)",
    "Usual Suspects",
  ];
  @override
  List<Widget>? buildActions(BuildContext context) {
    // TODO: implement buildActions
    return [
      IconButton(
          onPressed: () {
            query = '';
          },
          icon: const Icon(Icons.clear))
    ];
  }

  @override
  Widget? buildLeading(BuildContext context) {
    // TODO: implement buildLeading
    // throw UnimplementedError();
    return IconButton(
        onPressed: () {
          close(context, null);
        },
        icon: const Icon(Icons.arrow_back));
  }

  @override
  Widget buildResults(BuildContext context) {
    // TODO: implement buildResults
    // throw UnimplementedError();
    List<String> matchQuery = [];
    for (var item in searchTerm) {
      if (item.toLowerCase().contains(query.toLowerCase())) {
        matchQuery.add(item);
      }
    }
    return ListView.builder(
      itemCount: matchQuery.length,
      itemBuilder: (context, index) {
        var result = matchQuery[index];
        return ListTile(
          title: Text(result),
        );
      },
    );
  }

  @override
  Widget buildSuggestions(BuildContext context) {
    // TODO: implement buildSuggestions
    // throw UnimplementedError();
    List<String> matchQuery = [];
    for (var item in searchTerm) {
      if (item.toLowerCase().contains(query.toLowerCase())) {
        matchQuery.add(item);
      }
    }
    return ListView.builder(
      itemCount: matchQuery.length,
      itemBuilder: (context, index) {
        var result = matchQuery[index];
        return ListTile(
          title: Text(result),
        );
      },
    );
  }
}
