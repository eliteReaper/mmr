import 'dart:developer';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:mmr_app/main.dart';
import 'package:mmr_app/models/user.dart';

import 'Pages/account_page.dart';
import 'Pages/game_page.dart';
import 'Pages/movie_page.dart';
import 'Pages/music_page.dart';

class TempScreen extends StatelessWidget {
  const TempScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    Navigator.popAndPushNamed(context, MyApp.Main_Screen);
    // log("Came here");
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
  static const recommend_list = ["Movie", "Profile"];
  final screens = [
    const MoviePage(),
    // const MusicPage(),
    // const GamePage(),
    AccountPage(),
  ];
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      floatingActionButton: Visibility(
        visible: isVisible,
        child: FloatingActionButton(
          elevation: 2,
          backgroundColor: color_on_index,
          tooltip: "Recommend " + recommend_list[currentIndex],
          onPressed: () {
            DatabaseReference ref = FirebaseDatabase.instance.ref('movies');
            var user = (FirebaseAuth.instance.currentUser!.uid.toString());
            List<CustomUser> user_list = [];
            ref.child(user).once().then(
              (snapshot) {
                Map<String, String> mp = Map();
                for (var snap in snapshot.snapshot.children) {
                  // print(snap.key.toString());
                  // print(snap.value.toString());
                  for (var child in snap.children) {
                    // print(child.key! + " -- " + child.value.toString());
                    mp[child.key!] = child.value.toString();
                  }
                  String? name = mp['name'];
                  String? rating = mp['rating'];
                  String? time = mp['time'];
                  user_list.add(CustomUser(user, name!, rating!, time!));
                }
              },
            ).whenComplete(() {
              // print(user_list.toString());
              if (user_list.length < 5) {
                showDialog(
                    context: context,
                    builder: (BuildContext context) {
                      return AlertDialog(
                        title: const Text("Insufficient data"),
                        content: const Text(
                            "Please rate atleast 5 movies to get the recommendations"),
                        actions: [
                          TextButton(
                            onPressed: () {
                              Navigator.of(context, rootNavigator: true)
                                  .pop("Discard");
                            },
                            child: const Text('Okay'),
                          )
                        ],
                      );
                    });
              } else {}
            });
            // TODO: Call the recommender system to provide with the recommendation based on the index for the page :)
          },
          child: const Icon(Icons.recommend),
        ),
      ),
      body: screens[currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: currentIndex,
        onTap: (index) {
          setState(() {
            currentIndex = index;
            if (index == 0) {
              color_on_index = Colors.red;
              isVisible = true;
              // } else if(index == 1) {
              //   color_on_index = Colors.amber;
              //   isVisible = true;
              // } else if(index == 2) {
              //   color_on_index = Colors.green;
              //   isVisible = true;
            } else {
              color_on_index = Colors.blue;
              isVisible = false;
            }
          });
        },
        items: const [
          BottomNavigationBarItem(
              icon: Icon(Icons.movie),
              label: 'Movies',
              backgroundColor: Colors.red),
          // BottomNavigationBarItem(icon: Icon(CupertinoIcons.music_albums), label: 'Music', backgroundColor: Colors.amber),
          // BottomNavigationBarItem(icon: Icon(CupertinoIcons.game_controller), label: 'Games', backgroundColor: Colors.green),
          BottomNavigationBarItem(
              icon: Icon(CupertinoIcons.person),
              label: 'Account',
              backgroundColor: Colors.blue),
        ],
      ),
    );
  }
}
