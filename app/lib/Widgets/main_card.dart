import 'package:flutter/cupertino.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_rating_bar/flutter_rating_bar.dart';
import 'package:flutter_slidable/flutter_slidable.dart';

// ignore: non_constant_identifier_names

// ignore: non_constant_identifier_names
Widget CustomSlidable(
    String name, String rating, String genre, BuildContext context, VoidCallback rateit, VoidCallback deleteit) {
  var ratings = 4.0;
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
          onPressed: (BuildContext context) => deleteit,
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
                      onPressed: rateit,
                    ),
                    TextButton(
                      child: const Text("Cancel"),
                      onPressed: () {
                        Navigator.of(context, rootNavigator: true).pop("Discard");
                      },
                    ),
                  ],
                  content: RatingBar.builder(
                    initialRating: 3,
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
          side: const BorderSide(color: Colors.black, width: 1),
          borderRadius: BorderRadius.circular(10),
        ),
        child: Container(
          padding: const EdgeInsets.all(5),
          child: Row(
            children: [
              Expanded(child: MovieDetails(name, genre)),
              RatingContainer(rating),
            ],
          ),
        )),
  );
}

// ignore: non_constant_identifier_names
Widget MovieName(String name) {
  return Text(
    name,
    style: const TextStyle(fontSize: 22),
    textAlign: TextAlign.left,
  );
}

// ignore: non_constant_identifier_names
Widget MovieGenre(String genre) {
  return Container(
    padding: const EdgeInsets.all(5),
    child: Text(
      genre,
      style: const TextStyle(fontSize: 18),
      textAlign: TextAlign.center,
      softWrap: true,
    ),
    decoration: BoxDecoration(
      borderRadius: const BorderRadius.all(Radius.circular(12)),
      border: Border.all(style: BorderStyle.solid, color: Colors.black26),
      color: Colors.black12,
    ),
  );
}

// ignore: non_constant_identifier_names
Widget MovieDetails(String name, String genre) {
  return Container(
    margin: const EdgeInsets.all(5),
    child: Column(
      mainAxisAlignment: MainAxisAlignment.start,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        MovieName(name),
        const SizedBox(
          height: 10,
        ),
        MovieGenre(genre)
      ],
    ),
  );
}

// ignore: non_constant_identifier_names
Widget RatingContainer(String rating) {
  var color = Colors.green[700];
  var value = double.parse(rating);
  if (value >= 4.0) {
    color = Colors.green[800];
  } else if (value >= 3.0) {
    color = Colors.green[500];
  } else if (value >= 2.0) {
    color = Colors.yellow;
  } else if (value >= 1.0) {
    color = Colors.orange;
  } else {
    color = Colors.red;
  }
  return Container(
    width: 70,
    height: 40,
    padding: const EdgeInsets.all(8),
    alignment: Alignment.center,
    decoration: BoxDecoration(
      borderRadius: BorderRadius.circular(10),
      color: color,
    ),
    child: Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          rating,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(width: 10),
        const Icon(
          Icons.star,
          color: Colors.white,
          size: 20,
        )
      ],
    ),
  );
}
