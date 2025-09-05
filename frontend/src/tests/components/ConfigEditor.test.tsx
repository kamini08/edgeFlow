import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import ConfigEditor from '../../components/ConfigEditor';

describe('ConfigEditor', () => {
  test('calls onChange when edited', () => {
    const onChange = jest.fn();
    const { getByDisplayValue } = render(
      <ConfigEditor initialContent={'a=1'} onChange={onChange} />
    );
    const textarea = document.querySelector('textarea') as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: 'b=2' } });
    expect(onChange).toHaveBeenCalledWith('b=2');
  });
});

