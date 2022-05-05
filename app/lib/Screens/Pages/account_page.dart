import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:mmr_app/main.dart';
import 'package:mmr_app/screens/rated_data_page.dart';

import '../../models/user.dart';
import '../../widgets/widget.dart';

class AccountPage extends StatefulWidget {
  @override
  _AccountPageState createState() => _AccountPageState();
}

class _AccountPageState extends State<AccountPage> {
  List<CustomUser> user_list = [];

  List < CustomUser > returnList(){
    return user_list;
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        elevation: 0.0,
        backgroundColor: const Color(0xff555555),
      ),
      body: Stack(
        // crossAxisAlignment: CrossAxisAlignment.center,
        alignment: Alignment.center,
        children: [
          CustomPaint(
            child: Container(
              width: MediaQuery.of(context).size.width,
              height: MediaQuery.of(context).size.height,
            ),
            painter: HeaderCurvedContainer(),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const Padding(
                padding: EdgeInsets.all(20),
                child: Text(
                  "Profile",
                  style: TextStyle(
                    fontSize: 35,
                    letterSpacing: 1.5,
                    color: Colors.white,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.all(10.0),
                width: MediaQuery.of(context).size.width / 2,
                height: MediaQuery.of(context).size.width / 2,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.white, width: 5),
                  shape: BoxShape.circle,
                  color: Colors.white,
                  image: const DecorationImage(
                    fit: BoxFit.cover,
                    image: AssetImage('images/download.jpg'),
                  ),
                ),
              ),
            ],
          ),
          Padding(
            padding: const EdgeInsets.only(bottom: 270, left: 184),
            child: CircleAvatar(
              backgroundColor: Colors.black54,
              child: IconButton(
                icon: const Icon(
                  Icons.edit,
                  color: Colors.white,
                ),
                onPressed: () {},
              ),
            ),
          ),
          Column(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              Container(
                height: 450,
                width: double.infinity,
                margin: EdgeInsets.symmetric(horizontal: 10),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Padding(
                      padding: EdgeInsets.symmetric(horizontal: 10),
                      child: custombox(
                        context,
                        FirebaseAuth.instance.currentUser!.displayName
                            .toString(),
                        const Icon(FontAwesomeIcons.user),
                      ),
                    ),
                    const SizedBox(
                      height: 15,
                    ),
                    Padding(
                      padding: EdgeInsets.symmetric(horizontal: 10),
                      child: custombox(
                        context,
                        FirebaseAuth.instance.currentUser!.email.toString(),
                        const Icon(Icons.email),
                      ),
                    ),
                    const SizedBox(
                      height: 15,
                    ),
                    FlatButton(
                      onPressed: () {
                        // print("Pressed");
                        DatabaseReference ref =
                            FirebaseDatabase.instance.ref('movies');
                        var user =
                            (FirebaseAuth.instance.currentUser!.uid.toString());
                        user_list.clear();
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
                              user_list
                                  .add(CustomUser(user, name!, rating!, time!));
                            }
                          },
                        ).whenComplete(
                          () {
                            print("Pressed");
                            if (user_list.isEmpty) {
                              print("user dialog");
                              showDialog(
                                  context: context,
                                  builder: (BuildContext context) {
                                    return AlertDialog(
                                      title: const Text("No data"),
                                      content: const Text(
                                          "No rating data can be found for the specified user"),
                                      actions: [
                                        TextButton(
                                          onPressed: () {
                                            Navigator.of(context,
                                                    rootNavigator: true)
                                                .pop("Discard");
                                          },
                                          child: const Text("Okay"),
                                        ),
                                      ],
                                    );
                                  });
                            } else {
                              print("kidharr kamoo");
                              Navigator.pushNamed(context, MyApp.Rated_data);
                            }
                          },
                        );
                      },
                      child: custombox(
                        context,
                        "Media you have rated",
                        Icon(Icons.star),
                      ),
                    ),
                    const SizedBox(
                      height: 15,
                    ),
                    // logout
                    FlatButton(
                      onPressed: () {
                        FirebaseAuth.instance.signOut();
                        Navigator.popAndPushNamed(context, MyApp.Login_Screen);
                      },
                      child: custombox(
                        context,
                        "Logout",
                        const Icon(Icons.logout),
                      ),
                    ),
                    const SizedBox(
                      height: 15,
                    ),
                  ],
                ),
              )
            ],
          ),
        ],
      ),
    );
  }
}

class HeaderCurvedContainer extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    Paint paint = Paint()..color = const Color(0xff555555);
    Path path = Path()
      ..relativeLineTo(0, 150)
      ..quadraticBezierTo(size.width / 2, 225, size.width, 150)
      ..relativeLineTo(0, -150)
      ..close();
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
