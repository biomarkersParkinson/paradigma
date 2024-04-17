%% disp error between two matrices ..
function dispEE( a, b );
 
ee = a - b;
diff_re = max( abs(real(ee(:))) );
diff_im = max( abs(imag(ee(:))) );
a_name=deblank(inputname(1));
b_name=deblank(inputname(2));
fprintf('<< %s - %s = %g + j%g >>\n',a_name,b_name,diff_re,diff_im);
