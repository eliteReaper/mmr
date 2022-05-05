import 'package:flutter/cupertino.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_rating_bar/flutter_rating_bar.dart';
import 'package:flutter_slidable/flutter_slidable.dart';

// ignore: non_constant_identifier_names
Widget MovieName(String name) {
  return Text(
    name,
    style: const TextStyle(fontSize: 22),
    textAlign: TextAlign.left,
  );
}

Widget MovieGenresList(List<String> genre) {
  return Container(
    child: ListView.builder(
      itemBuilder: (context, index) {
        return Container(
          padding: const EdgeInsets.all(5),
          child: Text(
            genre[index],
            style: const TextStyle(fontSize: 14),
            textAlign: TextAlign.center,
            softWrap: true,
          ),
          decoration: BoxDecoration(
            borderRadius: const BorderRadius.all(Radius.circular(12)),
            border: Border.all(style: BorderStyle.solid, color: Colors.black26),
            color: Colors.black12,
          ),
        );
      },
    ),
  );
}

// ignore: non_constant_identifier_names
Widget MovieGenre(String genre) {
  return Container(
    padding: const EdgeInsets.all(5),
    child: Text(
      genre,
      style: const TextStyle(fontSize: 14),
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
