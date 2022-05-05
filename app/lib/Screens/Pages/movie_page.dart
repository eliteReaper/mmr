import 'dart:io';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_rating_bar/flutter_rating_bar.dart';
import 'package:flutter_slidable/flutter_slidable.dart';
import 'package:mmr_app/constants/moviedataconstant.dart';

import '../../widgets/main_card.dart';

class MoviePage extends StatefulWidget {
  const MoviePage({Key? key}) : super(key: key);
  static List<String> movies = movie;
  static List<String> genres = genre;

  @override
  State<MoviePage> createState() => _MoviePageState();
}

class _MoviePageState extends State<MoviePage> {
  @override
  Widget build(BuildContext context) {
    var url = FirebaseDatabase.instance.databaseURL;
    // print("firebase url: $url");
    // FirebaseDatabase.instance.reference().child('student').set({
    //   'name':'ishaan',
    //   'class':'cs-a-3',
    // });
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
        child: ListView.builder(
          itemCount: genre.length,
          itemBuilder: (context, index) {
            final name = movie[index];
            final gen = genre[index];
            return CustomSlidable(name, "2.2", gen, context);
          },
        ),
      ),
    );
  }
}

// ignore: non_constant_identifier_names
Widget CustomSlidable(
    String name, String rating, String genre, BuildContext context) {
  var ratings = 4.0;
  final File file = File('./');
  return Slidable(
    // Specify a key if the Slidable is dismissible.
    key: const ValueKey(0),
    // The start action pane is the one at the left or the top side.
    startActionPane: ActionPane(
      // A motion is a widget used to control how the pane animates.
      motion: const ScrollMotion(),
      // A pane can dismiss the Slidable.
      dismissible: DismissiblePane(onDismissed: () {}),
      // All actions are defined in the children parameter.
      children: [
        // A SlidableAction can have an icon and/or a label.
        SlidableAction(
          onPressed: (BuildContext context) => () {
            // TODO implement them :)
          },
          backgroundColor: const Color(0xFFFE4A49),
          foregroundColor: Colors.white,
          icon: Icons.delete,
          label: 'Delete',
        ),
      ],
    ),
    // The end action pane is the one at the right or the bottom side.
    endActionPane: ActionPane(
      motion: const ScrollMotion(),
      children: [
        SlidableAction(
          onPressed: (BuildContext context) {
            showDialog(
              context: context,
              builder: (BuildContext context) {
                return AlertDialog(
                  title: const Text("Rate the movie\n"),
                  actions: [
                    TextButton(
                      child: const Text("Rate"),
                      onPressed: () {
                        var userID =
                            FirebaseAuth.instance.currentUser!.uid.toString();
                        DatabaseReference ref =
                            FirebaseDatabase.instance.ref('movies');
                        Map<String, String> mp = {};
                        mp['name'] = name;
                        mp['rating'] = ratings.toString();
                        mp['time'] =
                            DateTime.now().millisecondsSinceEpoch.toString();
                        ref.child(userID).child(name).set(mp).asStream();

                        print('The value of the input is: $userID');
                        Navigator.of(context, rootNavigator: true)
                            .pop("Discard");
                      },
                    ),
                    TextButton(
                      child: const Text("Cancel"),
                      onPressed: () {
                        Navigator.of(context, rootNavigator: true)
                            .pop("Discard");
                      },
                    ),
                  ],
                  content: RatingBar.builder(
                    initialRating: 4.0,
                    minRating: 1,
                    direction: Axis.horizontal,
                    allowHalfRating: true,
                    itemCount: 5,
                    itemPadding: const EdgeInsets.symmetric(horizontal: 2.0),
                    itemBuilder: (context, _) => const Icon(
                      Icons.star,
                      color: Colors.amber,
                    ),
                    onRatingUpdate: (rating) {
                      ratings = rating;
                      if (kDebugMode) {
                        print(rating);
                      }
                    },
                  ),
                );
              },
            );
          },
          backgroundColor: const Color(0xFF7BC043),
          foregroundColor: Colors.white,
          icon: Icons.stars,
          label: 'Rate',
        ),
      ],
    ),
    // The child of the Slidable is what the user sees when the
    // component is not dragged.
    child: Card(
        elevation: 10,
        shape: RoundedRectangleBorder(
          // side: const BorderSide(color: Colors.black, width: 1),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Container(
          padding: const EdgeInsets.all(5),
          child: Row(
            children: [
              Expanded(child: MovieDetails(name, genre)),
              Padding(
                child: RatingContainer(rating),
                padding: EdgeInsets.only(right: 10),
              ),
            ],
          ),
        )),
  );
}

class CustomSearchDelegate extends SearchDelegate {
  List<String> searchTerm = movie;
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
