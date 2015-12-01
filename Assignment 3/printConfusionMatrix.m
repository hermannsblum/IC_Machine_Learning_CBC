function [  ] = printConfusionMatrix( confMatrix )

disp(['True\backslash Predicted & ' emolab2str(1) ' & ' emolab2str(2) ' & ' emolab2str(3) ' & ' emolab2str(4) ' & ' emolab2str(5) ' & ' emolab2str(6) ' \\ ']);
for i=1:6
    row = emolab2str(i);
    for j=1:6
        row = [row ' & ' num2str(confMatrix(i,j))];
    end
    disp([row ' \\']);

end

