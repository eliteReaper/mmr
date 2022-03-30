import 'dart:developer';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:mmr_app/Screens/Pages/account_page.dart';
import 'package:mmr_app/Screens/Pages/game_page.dart';
import 'package:mmr_app/Screens/Pages/movie_page.dart';
import 'package:mmr_app/Screens/Pages/music_page.dart';
import 'package:mmr_app/main.dart';


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
  final screens = [
    MoviePage(),
    MusicPage(),
    GamePage(),
    AccountPage(),
  ];
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: screens[currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: currentIndex,
        onTap: (index){
          setState(() {
            currentIndex = index;
          });
        },
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.movie), label: 'Movies', backgroundColor: Colors.red),
          BottomNavigationBarItem(icon: Icon(CupertinoIcons.music_albums), label: 'Music', backgroundColor: Colors.amber),
          BottomNavigationBarItem(icon: Icon(CupertinoIcons.game_controller), label: 'Games', backgroundColor: Colors.green),
          BottomNavigationBarItem(icon: Icon(FontAwesomeIcons.user), label: 'Account', backgroundColor: Colors.blue),
        ],
      ),
    );
  }
}
