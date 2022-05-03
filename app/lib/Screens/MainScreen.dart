import 'dart:developer';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:mmr_app/main.dart';

import 'Pages/account_page.dart';
import 'Pages/game_page.dart';
import 'Pages/movie_page.dart';
import 'Pages/music_page.dart';


class TempScreen extends StatelessWidget {
  const TempScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    Navigator.popAndPushNamed(context, MyApp.Main_Screen);
    log("Came here");
    return Container();
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({Key? key}) : super(key: key);

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int currentIndex = 0;
  // ignore: non_constant_identifier_names
  var color_on_index = Colors.red;
  var isVisible = true;
  // ignore: constant_identifier_names
  static const recommend_list = ["Movie", "Music", "Game", "Profile"];
  final screens = [
    const MoviePage(),
    const MusicPage(),
    const GamePage(),
    const AccountPage(),
  ];
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      floatingActionButton: Visibility(
        visible: isVisible,
        child: FloatingActionButton(
          elevation: 2,
          backgroundColor: color_on_index,
          tooltip: "Recommend "+recommend_list[currentIndex],
          onPressed: () {
            // TODO: Call the recommender system to provide with the recommendation based on the index for the page :)
          },
          child: const Icon(Icons.recommend),

        ),
      ),
      body: screens[currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: currentIndex,
        onTap: (index){
          setState(() {
            currentIndex = index;
            if(index == 0) {
              color_on_index = Colors.red;
              isVisible = true;
            } else if(index == 1) {
              color_on_index = Colors.amber;
              isVisible = true;
            } else if(index == 2) {
              color_on_index = Colors.green;
              isVisible = true;
            } else {
              color_on_index = Colors.blue;
              isVisible = false;
            }
          });
        },
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.movie), label: 'Movies', backgroundColor: Colors.red),
          BottomNavigationBarItem(icon: Icon(CupertinoIcons.music_albums), label: 'Music', backgroundColor: Colors.amber),
          BottomNavigationBarItem(icon: Icon(CupertinoIcons.game_controller), label: 'Games', backgroundColor: Colors.green),
          BottomNavigationBarItem(icon: Icon(CupertinoIcons.person), label: 'Account', backgroundColor: Colors.blue),
        ],
      ),
    );
  }
}
