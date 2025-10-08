i m p o r t n u m p y as np i m p o r t m a t p l o t l i b . p y p l o t as p l t
2
3 # %% p o u r la f o n c t i o n c o s
4 a==10
5 b=10
6 fe=3
7 N=i n t ( ( b=a ) / ( 2* np . pi ) * fe )
8 k T e=np . l i n s p a c e ( a , b , N )
9 e c h=np . c o s ( k T e )
10 Te=k T e [1 ] = k T e [ 0 ]
11
12 d e f s i n c ( x ) :
13 if x==0 :
14 r e t u r n 1
15 e l s e :
16 r e t u r n np . s i n ( x ) /x
17
18 v e c s i n c=np . v e c t o r i z e ( s i n c )
19
20 d e f r e c o n s t r u ( t , ech , kTe , Te ) :
21 l=l e n ( t )
22 s=np . z e r o s ( l )
23 i n d s o m m e=l e n ( e c h )
24 f o r i in r a n g e ( l ) :
25 st=0
26 f o r k in r a n g e ( i n d s o m m e ) :
27 st+=e c h [ k ] * v e c s i n c ( np . pi *( t [ i]= k T e [ k ] ) / Te )
28 s [ i]= st
29 r e t u r n s
30
31 t=np . l i n s p a c e ( a , b , 1 0 0 0 )
32 s i g n a l _ r e c o n s= r e c o n s t r u ( t , ech , kTe , Te )
33
34 fig , ax = p l t . s u b p l o t s ( 2 , f i g s i z e =(15 , 7 ) )
35 f i g . s u p t i t l e ( " T D 3 : E x e r c i c e 1 : t h é o r è m e d ' é c h a n t i l l o n n a g e " )
36 ax [ 0 ] . s e t _ t i t l e ( f " E c h a n t i l l o n n a g e p o u r fe ={ fe } " )
37 ax [ 0 ] . p l o t ( t , np . c o s ( t ) , 'b ' , l a b e l=" s i g n a l o r i g i n e l " )
38 ax [ 0 ] . p l o t ( t , s i g n a l _ r e c o n s , ' -r ' , l a b e l=" s i g n a l r e c o n s t r u i t " )
39 ax [ 0 ] . p l o t ( kTe , ech , ' og ' , l a b e l=" p o i n t s é c h a n t i l l o n n é s " )
40 ax [ 0 ] . l e g e n d ( )
41
42 d e f TF ( a l p h a ) :
43 s=10*( v e c s i n c (10*(1 = a l p h a ) ) +v e c s i n c (10*(1+ a l p h a ) ) )
44 r e t u r n s
45
46 a l p h a=np . l i n s p a c e ( =50 ,50 , 1 0 0 0 0 )
ax [ 1 ] . s e t _ t i t l e ( " G r a p h e de la t r a n s f o r m é e de F o u r i e r " )
ax [ 1 ] . a x v s p a n (=2* np . pi *fe , 2* np . pi *fe , a l p h a =0 .5 , c o l o r=' r e d ' , l a b e l=" p l a g e d e s f r é q u e n c e s c o n s e r v é e s " )
ax [ 1 ] . p l o t ( a l p h a , TF ( a l p h a ) , 'g ' )
ax [ 1 ] . l e g e n d ( )
